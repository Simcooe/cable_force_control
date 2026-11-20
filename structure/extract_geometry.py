#!/usr/bin/env python3
"""Read-only geometry extraction for the active four_cord_sim assembly.

Run from any directory with four_cord_sim/.venv/bin/python -B extract_geometry.py.
Only the three requested deliverables and geometry_attachments/ are written.
No integration, controller, parameter fitting, or source-model rebuild is run.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'four_cord_sim/model'
OUT = ROOT / 'geometry_attachments'
sys.path.insert(0, str(MODEL))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from drive_geometry import BEND, SLIDE, DriveGeometry
from incline_press_model import build_incline_press, BEND_SHAFTS, SHAFTS
from whole_guide_contact import add_whole_guides, guide_specs, ROUTE_TO_LABEL
from whole_native_cords import add_native_cords, bind_native_cords
from r2_contact_fixture import AXIAL_RIGIDITY_N, INITIAL_TENSION_N, DRAG_PER_S
from r2_groove_contact import wheel_profile

STATUS_XML = 'XML明确值'
STATUS_RUNTIME = '编译/运行时读取'
STATUS_DERIVED = '几何推算'
STATUS_MISSING = '缺失'


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serial,
                               allow_nan=False) + '\n')


def src(path, obj, status):
    path = Path(path)
    return dict(file=str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
                object=obj, status=status)


def pose(p, r):
    return dict(position_m=np.asarray(p) / 1000,
                quaternion_wxyz=Rotation.from_matrix(np.asarray(r).reshape(3, 3)).as_quat(scalar_first=True),
                rotation_matrix=np.asarray(r).reshape(3, 3))


def nearest_triangle(point, vertices, faces):
    """Closest point to an unmodified tessellated surface (not its convex hull)."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    nn = np.einsum('ij,ij->i', normal, normal)
    ap = point-a
    projection = point - normal * (np.einsum('ij,ij->i', ap, normal) /
                                   np.maximum(nn, 1.e-60))[:, None]
    v = projection-a
    aa = np.einsum('ij,ij->i', ab, ab)
    bb = np.einsum('ij,ij->i', ab, ac)
    cc = np.einsum('ij,ij->i', ac, ac)
    va = np.einsum('ij,ij->i', v, ab)
    vc = np.einsum('ij,ij->i', v, ac)
    det = np.maximum(aa*cc-bb*bb, 1.e-60)
    u, w = (cc*va-bb*vc)/det, (aa*vc-bb*va)/det
    inside = (nn > 1.e-40) & (u >= 0) & (w >= 0) & (u+w <= 1)
    candidates = [projection[inside]]
    for p0, p1 in ((a, b), (b, c), (c, a)):
        delta = p1-p0
        t = np.clip(np.einsum('ij,ij->i', point-p0, delta) /
                    np.maximum(np.einsum('ij,ij->i', delta, delta), 1.e-60), 0, 1)
        candidates.append(p0 + t[:, None]*delta)
    pts = np.vstack(candidates)
    k = int(np.argmin(np.linalg.norm(pts-point, axis=1)))
    return float(np.linalg.norm(pts[k]-point)), pts[k]


def raw_mesh_bounds(path):
    if path.suffix.lower()=='.stl':
        raw=path.read_bytes();n=int.from_bytes(raw[80:84],'little')
        if len(raw)!=84+50*n: return None
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
        vertices=np.frombuffer(raw,dtype=dtype,offset=84,count=n)['vertices'].reshape(-1,3)
    elif path.suffix.lower()=='.obj':
        vertices=np.array([[float(v) for v in line.split()[1:4]]
                           for line in path.read_text().splitlines() if line.startswith('v ')])
    else: return None
    return np.array([vertices.min(axis=0),vertices.max(axis=0)])


def build():
    config = read_json(ROOT / 'four_cord_sim/experiment.json')
    dt = 1.5e-6 if config['cord'] == 'r2' else .3e-6
    # Same geometry-building calls/defaults as position_trial.run_position_trial.
    def components(tree):
        tree.find('option').set('jacobian', 'sparse')
        material = (dict(axial_rigidity_N=AXIAL_RIGIDITY_N, pretension_N=INITIAL_TENSION_N,
                         drag_per_s=DRAG_PER_S) if config['cord'] == 'r2' else {})
        return dict(guides=add_whole_guides(tree, sectors=config.get('sectors', 48), four_cords=True),
                    cords=add_native_cords(tree, spacing_mm=.30, four_cords=True,
                        orange_slack_mm=config.get('orange_slack_mm', .20),
                        purple_slack_mm=config.get('purple_slack_mm', 0.), **material))
    model, data, spec = build_incline_press(tilt_deg=config['tilt_deg'], dt=dt,
        contact=config.get('plate', True), native_setup=components, four_cords=True,
        shared_axial_shaft=True, independent_bend=True,
        solid47_solid46_collision=config.get('solid47_solid46_collision', False),
        motor_armature_kg_m2=config.get('motor_armature_kg_m2'),
        carriage_solid46_collision=config.get('carriage_solid46_collision', False),
        carriage_solid46_friction=config.get('carriage_solid46_friction', .20),
        carriage_direct_contact=config.get('carriage_friction_mode') == 'direct_contact')
    mujoco.mj_forward(model, data)
    return config, model, data, spec


def extract():
    OUT.mkdir(exist_ok=True)
    original_files = [p for p in (ROOT/'four_cord_sim').rglob('*')
                      if p.is_file() and (p.parent == ROOT/'four_cord_sim' or
                         MODEL in p.parents) and p.suffix in ('.py', '.xml', '.json')
                      and 'results' not in p.parts and 'assets' not in p.parts]
    originals_before = {str(p.relative_to(ROOT)): sha(p) for p in original_files}
    config, m, d, spec = build()
    tree = ET.fromstring(spec['native_xml'])
    (OUT/'effective_model.xml').write_text(spec['native_xml'])
    cords = bind_native_cords(m, spec['native']['cords'])['cords']
    candidate = read_json(MODEL/'data/corrected_routes_candidate.json')
    bg = read_json(MODEL/'data/ball_geometry.json')
    manifest = read_json(MODEL/'data/cad_manifest.json')
    source_graph = read_json(ROOT/'mujoco_migration/data/source_graph.json')
    rigid = read_json(MODEL/'data/rigid_validation.json')
    xbody = {b.get('name'): b for b in tree.iter('body')}
    xsite = {s.get('name'): s for s in tree.iter('site') if s.get('name')}
    xgeom = {g.get('name'): g for g in tree.iter('geom') if g.get('name')}
    xmesh = {a.get('name'): a for a in tree.findall('asset/mesh')}
    joint_xml = {j.get('name'): j for j in tree.iter('joint') if j.get('name')}

    def bodyname(bid): return m.body(int(bid)).name or ('world' if bid == 0 else None)
    def site_record(name):
        i = m.site(name).id
        return dict(object=name, body=bodyname(m.site_bodyid[i]),
                    local_frame='所属body', world_frame='world',
                    local_position_m=m.site_pos[i]/1000,
                    local_quaternion_wxyz=m.site_quat[i].copy(),
                    world_pose=pose(d.site_xpos[i], d.site_xmat[i]),
                    xml_attributes=dict(xsite[name].attrib),
                    provenance=[src('geometry_attachments/effective_model.xml', name, STATUS_XML),
                                src('geometry_attachments/effective_model.xml', 'mj_forward: '+name, STATUS_RUNTIME)])

    def mesh_world(gid):
        mid = int(m.geom_dataid[gid]); a = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts = d.geom_xpos[gid]/1000 + (m.mesh_vert[a:a+n]/1000) @ d.geom_xmat[gid].reshape(3, 3).T
        return verts, m.mesh_face[fa:fa+fn]

    # All mechanism nodes, including fixed helper bodies and required ancestors.
    bodies = []
    for i in range(m.nbody):
        name = bodyname(i)
        if name and name.startswith('native_') and '_particle_' in name:
            continue
        ids = np.flatnonzero(m.jnt_bodyid == i)
        bodies.append(dict(id=i, name=name, parent=bodyname(m.body_parentid[i]) if i else None,
            weld_group_id=int(m.body_weldid[i]), local_position_m=m.body_pos[i]/1000,
            local_quaternion_wxyz=m.body_quat[i].copy(),
            world_pose=pose(d.xpos[i], d.xmat[i]),
            joints=[m.joint(int(j)).name for j in ids],
            connection='world' if i == 0 else ('关节连接' if len(ids) else '父子固定连接（无joint）'),
            source=src('geometry_attachments/effective_model.xml', 'body:'+str(name), STATUS_RUNTIME)))

    joints, layout = [], []
    qpos_si, qvel_si, qpos0_si = d.qpos.copy(), d.qvel.copy(), m.qpos0.copy()
    kind = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
    for i in range(m.njnt):
        typ = kind[int(m.jnt_type[i])]
        qa, va = int(m.jnt_qposadr[i]), int(m.jnt_dofadr[i])
        nq, nv = {'free': (7, 6), 'ball': (4, 3), 'slide': (1, 1), 'hinge': (1, 1)}[typ]
        if typ == 'slide':
            qpos_si[qa:qa+nq] /= 1000; qpos0_si[qa:qa+nq] /= 1000; qvel_si[va:va+nv] /= 1000
        name, bid = m.joint(i).name, int(m.jnt_bodyid[i])
        rec = dict(id=i, name=name, display_name=name or f'unnamed_joint_id_{i}', type=typ,
            body=bodyname(bid), parent=bodyname(m.body_parentid[bid]),
            qpos_address=qa, qpos_dimension=nq, qvel_address=va, qvel_dimension=nv,
            qpos_unit='m' if typ == 'slide' else ('quaternion_wxyz' if typ == 'ball' else 'rad'),
            qvel_unit='m/s' if typ == 'slide' else 'rad/s')
        layout.append(rec.copy())
        if not name:
            continue
        spring = m.qpos_spring[qa:qa+nq].copy()
        if typ == 'slide': spring /= 1000
        rec.update(local_axis_point_m=m.jnt_pos[i]/1000,
            local_axis_direction=None if typ == 'ball' else m.jnt_axis[i].copy(),
            world_axis_point_m=d.xanchor[i]/1000,
            world_axis_direction=None if typ == 'ball' else d.xaxis[i].copy(),
            ball_center_world_m=d.xanchor[i]/1000 if typ == 'ball' else None,
            axis_frame='所属body；world轴为当前参考姿态的mj_forward输出',
            qpos0=qpos0_si[qa:qa+nq].copy(), reference_qpos=qpos_si[qa:qa+nq].copy(),
            spring_reference=spring,
            limited=bool(m.jnt_limited[i]),
            range=(m.jnt_range[i]/(1000 if typ == 'slide' else 1)).copy() if m.jnt_limited[i] else None,
            reference_convention='ref未写时为MuJoCo默认0；ball零位为单位四元数；springref不是几何零位',
            xml_attributes=dict(joint_xml[name].attrib),
            source_joint_mapping=next((j for j in rigid['joint_mapping'] if name == 'joint_'+j['sid']), None),
            provenance=[src('geometry_attachments/effective_model.xml', name, STATUS_XML),
                        src('geometry_attachments/effective_model.xml', 'mj_forward '+name, STATUS_RUNTIME)])
        joints.append(rec)

    source_blocks = {str(b['sid']): b for b in source_graph['blocks']}
    relevant_sids = ['16757', '16758', '16762', '16763', '16764', '16765', '16766', '16767',
                     '16769', '16759', '16761', '16760', '16768', '16621', '16622', '16623',
                     '16624', '16625', '16727', '17387', '17389']
    parts = []
    for sid in relevant_sids:
        name = source_blocks[sid]['name']
        if sid == '16768': name = 'Solid5（旧别名Solid47，源SID17637）'
        gid = m.geom('solid_'+sid).id
        native = site_record('native_'+sid)
        mid = int(m.geom_dataid[gid]); meshname = m.mesh(mid).name
        cad = next((p for p in manifest if any(str(b['sid']) == sid for b in p['source_blocks'])), None)
        verts, faces = mesh_world(gid)
        rn = np.asarray(native['world_pose']['rotation_matrix']); pn = np.asarray(native['world_pose']['position_m'])
        vlocal = (verts-pn) @ rn
        bbox = np.array([vlocal.min(axis=0), vlocal.max(axis=0)])
        attrib = dict(xmesh[meshname].attrib)
        step_path = ROOT / cad['source_file'] if cad else None
        mesh_path = MODEL/'assets'/attrib['file'] if 'file' in attrib else None
        raw_bounds=raw_mesh_bounds(mesh_path) if mesh_path and mesh_path.exists() else None
        parts.append(dict(physical_part=name, source_sid=sid, simulation_sid=sid,
            actual_source_sid='17637' if sid=='16768' else sid,
            sid_note='source_sid为兼容源图SID；actual_source_sid为当前SLX块SID' if sid=='16768' else None,
            geom='solid_'+sid,
            native_frame=native, body=bodyname(m.geom_bodyid[gid]), fixed_with_body=True,
            xml_geom_attributes=dict(xgeom['solid_'+sid].attrib),
            compiled_geom_local_position_m=m.geom_pos[gid]/1000,
            compiled_geom_local_quaternion_wxyz=m.geom_quat[gid].copy(),
            compiled_geom_world_pose=pose(d.geom_xpos[gid], d.geom_xmat[gid]),
            mesh=dict(name=meshname, xml=attrib, compiled_recentering_position_m=m.mesh_pos[mid]/1000,
                compiled_recentering_quaternion_wxyz=m.mesh_quat[mid].copy(),
                compiled_vertices_back_in_native_frame_bounds_m=bbox,
                original_mesh_file_bounds_m=raw_bounds,
                original_mesh_vs_compiled_bounds_error_m=float(np.max(abs(bbox-raw_bounds))) if raw_bounds is not None else None,
                CAD_bounds_m=cad['bounds'] if cad else None,
                bounds_max_difference_m=float(np.max(abs(bbox-np.array(cad['bounds'])))) if cad else None,
                file_sha256=sha(mesh_path) if mesh_path and mesh_path.exists() else None,
                manifest_mesh_hash_matches=sha(mesh_path) == cad.get('mesh_sha256') if cad and mesh_path and mesh_path.exists() else None),
            source_step=dict(file=str(step_path.relative_to(ROOT)) if step_path else None,
                expected_sha256=cad.get('source_sha256') if cad else None,
                current_sha256=sha(step_path) if step_path and step_path.exists() else None),
            provenance=[src('geometry_attachments/effective_model.xml','solid_'+sid,STATUS_RUNTIME),
                        src(MODEL/'data/cad_manifest.json', 'source_blocks.sid='+sid, STATUS_DERIVED)]))
    pmap = {p['source_sid']: p for p in parts}

    anchors = []
    for color, label, sid, part_sid in [('blue','A1','16614','16767'),('green','A2','16613','16767'),
                                       ('orange','A3','16615','16757'),('purple','A4','16616','16757')]:
        s = site_record('route_'+sid)
        p = pmap[part_sid]['native_frame']['world_pose']
        point = np.asarray(s['world_pose']['position_m'])
        native_point = np.asarray(p['rotation_matrix']).T @ (point-np.asarray(p['position_m']))
        dist, near = nearest_triangle(point, *mesh_world(m.geom('solid_'+part_sid).id))
        anchors.append(dict(label=label,color=color,**s,physical_part=pmap[part_sid]['physical_part'],
            position_in_physical_CAD_frame_m=native_point,
            role='独立绳固定端' if color in ('blue','green') else '材料夹固点的模型端点（橙紫分成两条独立材料链）',
            physical_attachment_confirmed=False, inner_wall_or_hole_identity=None,
            surface_check=dict(method='当前编译后的原始视觉网格三角形最近点，非凸包',
                distance_m=dist,nearest_point_world_m=near,status=STATUS_DERIVED,
                conclusion='表面接近不能确认真实绳孔或夹固位置'),
            source_route=next(r['anchor_source'] for r in candidate['routes'] if r['spool_sid']==cords[color]['spool_sid'])))

    guide_records=[]
    profile, bore, _, _, _ = wheel_profile()
    for sid, g in guide_specs().items():
        physical=pmap[g['solid_sid']]
        native=physical['native_frame']; nr=np.asarray(native['world_pose']['rotation_matrix'])
        np0=np.asarray(native['world_pose']['position_m'])
        center=np0+nr@np.array(g['cad_center_mm'])/1000
        bid=m.body(physical['body']).id
        br=d.xmat[bid].reshape(3,3)
        contact_geoms=[m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.startswith('wc_guide_'+g['label']+'_')]
        aliases=[n for n in xsite if n.startswith('route_') and ROUTE_TO_LABEL.get(n[6:])==g['label']]
        guide_records.append(dict(label=g['label'],physical_part=physical['physical_part'],body=physical['body'],
            source_solid_sid=g['solid_sid'],route_station_sites=[site_record(n) for n in aliases],
            type='CAD有限圆柱' if g['kind']=='post' else 'CAD V槽回转轮，分扇区凸网格',
            center_local_body_m=br.T@(center-d.xpos[bid]/1000),center_world_m=center,
            axis_local_CAD=np.array([0.,1.,0.]),axis_local_body=br.T@nr@np.array([0.,1.,0.]),
            axis_world=nr@np.array([0.,1.,0.]),native_CAD_frame=native,
            physical_radius_m=g['surface_radius_mm']/1000,half_axial_length_m=g['half_length_mm']/1000,
            full_axial_length_m=2*g['half_length_mm']/1000,
            nominal_pitch_radius_m=next(n['radius_m'] for r in candidate['routes'] for n in r['nodes'] if n['sid']==sid),
            groove_profile_axial_position_radius_m=np.array(profile)/1000 if g['kind']=='groove' else None,
            groove_bore_radius_m=bore/1000 if g['kind']=='groove' else None,
            groove_flank_slope=g['slope'] if g['kind']=='groove' else None,
            collision_geoms=contact_geoms,
            follows_R=(physical['body']=='rigid_0008'), follows_X_k=False,
            provenance=[src(MODEL/'whole_guide_contact.py','guide_specs/add_whole_guides '+g['label'],STATUS_DERIVED),
                        src('geometry_attachments/effective_model.xml',contact_geoms[0],STATUS_RUNTIME)]))

    # Contact geometry only, no contact force evaluation or tension inference.
    contacts=[]
    for i,c in enumerate(d.contact):
        geom_names=[m.geom(int(g)).name if g>=0 else None for g in c.geom]
        flex_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_FLEX,int(f)) if f>=0 else None for f in c.flex]
        contacts.append(dict(index=i,geom_ids=np.array(c.geom).copy(),geoms=geom_names,
            flex_ids=np.array(c.flex).copy(),flexes=flex_names,flex_elements=np.array(c.elem).copy(),
            flex_vertices=np.array(c.vert).copy(),position_world_m=c.pos.copy()/1000,
            normal_world=c.frame[:3].copy(),contact_frame_rows=c.frame.reshape(3,3).copy(),
            signed_gap_m=float(c.dist)/1000,constraint_active=bool(c.efc_address>=0),
            provenance=STATUS_RUNTIME))
        rigid_sides=[]
        for gid in c.geom:
            if gid < 0: continue
            bid=int(m.geom_bodyid[gid]); rr=d.xmat[bid].reshape(3,3)
            rigid_sides.append(dict(geom=m.geom(int(gid)).name,body=bodyname(bid),
                contact_position_body_m=rr.T@(c.pos-d.xpos[bid])/1000,
                contact_normal_body=rr.T@c.frame[:3],
                normal_convention='仍沿原contact.frame第一行，不因转换body而翻转'))
        contacts[-1]['rigid_geom_sides']=rigid_sides
    write_json(OUT/'reference_contacts.json',dict(state='builder初始化+forward，t=0，无积分',
        convention='法向由MuJoCo contact.frame第一行给出，方向依赖geom/flex槽顺序；不等同于探头工作面法向',contacts=contacts))

    routes=[]
    color_order=('blue','green','orange','purple')
    for color, r in zip(color_order,candidate['routes'],strict=True):
        c=cords[color]; site_ids=c['site_ids']; points=d.site_xpos[site_ids]/1000
        nominal_guides=[]
        for g in r['guides']:
            rec={k:v for k,v in g.items() if k not in ('candidate_wrap_deg','signed_angle_XZ_deg','source_initial_wrap_deg')}
            rec.update(candidate_wrap_rad=math.radians(g['candidate_wrap_deg']),
                signed_angle_XZ_rad=math.radians(g['signed_angle_XZ_deg']),
                source_initial_wrap_rad=math.radians(g['source_initial_wrap_deg']),
                physical_label=ROUTE_TO_LABEL[g['sid']],status=STATUS_DERIVED)
            nominal_guides.append(rec)
        relevant_contacts=[ct['index'] for ct in contacts if 'native_'+color+'_edges' in ct['flexes']
                           and any(n and n.startswith('wc_guide_') for n in ct['geoms'])]
        contact_labels=sorted({n.split('_')[2] for i in relevant_contacts for n in contacts[i]['geoms'] if n and n.startswith('wc_guide_')})
        grouped={label:[] for label in contact_labels}
        for ci in relevant_contacts:
            ct=contacts[ci];slot=ct['flexes'].index('native_'+color+'_edges')
            for name in ct['geoms']:
                if name and name.startswith('wc_guide_'):
                    grouped[name.split('_')[2]].append((ci,int(ct['flex_elements'][slot])))
        current_order=[]
        for label,entries in grouped.items():
            ids=[ci for ci,_ in entries];elements=[e for _,e in entries]
            current_order.append(dict(label=label,body=next(g['body'] for g in guide_records if g['label']==label),
                first_material_edge=min(elements),last_material_edge=max(elements),contact_indices=ids,
                minimum_signed_gap_m=min(contacts[ci]['signed_gap_m'] for ci in ids),
                constraint_active_contact_count=sum(contacts[ci]['constraint_active'] for ci in ids),
                state='本次forward接触，未积分平衡；不保证为稳定物理绳路'))
        current_order.sort(key=lambda g:g['first_material_edge'])
        anchor=next(a for a in anchors if a['color']==color)
        direction=points[-2]-points[-1];direction/=np.linalg.norm(direction)
        anchor['toward_previous_material_point_world_unit']=direction
        anchor['toward_previous_material_point_body_unit']=d.xmat[m.body(anchor['body']).id].reshape(3,3).T@direction
        anchor['direction_provenance']=src('geometry_attachments/effective_model.xml','当前末段两site连线，不代表真实夹持切向',STATUS_DERIVED)
        routes.append(dict(color=color, anchor=next(a['label'] for a in anchors if a['color']==color),
            ordered_station_sites=[site_record('route_'+n['sid']) for n in r['nodes']],
            ordered_station_labels=[ROUTE_TO_LABEL.get(n['sid'], 'spool' if j==0 else next(a['label'] for a in anchors if a['color']==color))
                                    for j,n in enumerate(r['nodes'])],
            nominal_route=dict(status='工程中已有的投影候选支路，非当前材料链的强制约束',
                provenance=src(MODEL/'data/corrected_routes_candidate.json',color,STATUS_DERIVED),
                length_m=r['length_lifted_m'],spans=r['spans'],guides=nominal_guides,
                polyline_world_m=r['polyline_world_m'],extra_winding_turns=r['extra_winding_turns'],
                algorithm='world X-Z公切线+圆弧、直段按world Y差抬升；绕行符号以X-Z平面atan2定义，非各轮轴正向',
                active_guide_sids=r['active_guide_sids']),
            native_material_route=dict(status=STATUS_RUNTIME, flex='native_'+color+'_edges',
                endpoint_representation='首端world固定site；末端为刚体上的route锚点；内部粒子各3个平移DOF',
                ordered_sites=c['sites'],ordered_tendons=c['tendons'],
                site_bodies=[bodyname(m.site_bodyid[i]) for i in site_ids],
                ordered_world_points_m=points,
                tendon_lengths_m=d.ten_length[c['tendon_ids']].copy()/1000,
                total_tendon_length_m=float(d.ten_length[c['tendon_ids']].sum())/1000,
                sampled_polyline_length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                feed=site_record(c['sites'][0]),anchor_site=c['sites'][-1],
                initial_seed_length_m=c['reference_length_mm']/1000,
                guide_lane_m=c['guide_lane_mm']/1000,cord_radius_m=spec['native']['cords']['cord_radius_mm']/1000,
                current_guide_contact_labels=contact_labels,current_guide_contact_indices=relevant_contacts,
                current_contact_order_along_material_edges=current_order,
                contacts_file='geometry_attachments/reference_contacts.json',
                smooth_tangent_entry_exit_world_m=None,
                tangent_note='离散flex接触及节点弯折无唯一光滑切点；可用实际接触与节点。名义候选切点单独列出，不能替代当前切点。',
                seed_construction=src(MODEL/'whole_native_cords.py','_four_paths/_initial_polyline/_sample_path/add_native_cords',STATUS_DERIVED),
                seed_assumptions='蓝/绿Y车道±0.0005m；橙/紫局部Y抬升避开绿绳引线；不增加固定途经site，接触后可变'),
            material_continuity='蓝绿独立绳' if color in ('blue','green') else '设计：橙紫同一连续绳夹固于34；当前：两条不相连flex链，共用轴反向收放',
            branches='名义支路可随姿态脱离/接触；原生模型不通过固定wrap geom限定顺序，须读取当时接触'))

    spools=[]
    geom=DriveGeometry()
    for i,(color,r) in enumerate(zip(color_order,candidate['routes'],strict=True)):
        joint=geom.spool_joint_names[i]
        present=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,joint)>=0
        spools.append(dict(color=color,source_station='route_'+r['spool_sid'],station=site_record('route_'+r['spool_sid']),
            current_motor_joint=SHAFTS[0] if color=='purple' else joint,
            original_joint=joint,original_joint_present=present,
            pitch_radius_m=float(geom.r[geom.index[r['spool_sid']]]),
            payout_m_per_original_joint_rad=float(geom.payout_per_rad_m[i]),
            payout_m_per_current_joint_rad=float(geom.payout_per_rad_m[i])*(-1 if color=='purple' else 1),
            current_motor_joint_axis=next(j for j in joints if j['name']==(SHAFTS[0] if color=='purple' else joint)),
            source_spool_axis_world=r['nodes'][0]['source_axis_world'],
            actual_native_feed_site=cords[color]['sites'][0],
            source_model_angle_hint_rad=r['nodes'][0].get('source_spool_angle_rad'),
            physical_drum_groove_and_cable_attachment=None,
            implementation='薄环/route为源参考；固定出绳site+首段材料长度变更；不是接触卷绕在实体卷筒上',
            provenance=src(MODEL/'incline_press_model.py','make_shared_axial_shaft',STATUS_DERIVED)))

    # Re-express CAD-defined probe/guide features in the effective runtime frames.
    probe_face=site_record('probe_working_face')
    face=bg['probe_working_face']
    probe_axes=[]
    for label,record in bg['guide_geometry'].items():
        if not isinstance(record,dict): continue
        p=pmap[record['source_sid']]['native_frame']['world_pose']
        rr=np.asarray(p['rotation_matrix']); pp=np.asarray(p['position_m'])
        probe_axes.append(dict(label=label,body=pmap[record['source_sid']]['body'],
            local_CAD_point_m=record['point_native_m'],local_CAD_direction=record['direction_native'],
            world_point_m=pp+rr@record['point_native_m'],world_direction=rr@record['direction_native'],
            radius_m=record['radius_m'],source_faces=record['source_faces'],
            provenance=src(MODEL/'data/ball_geometry.json','guide_geometry.'+label,STATUS_DERIVED)))
    wall=m.geom('incline_surface').id; wr=d.geom_xmat[wall].reshape(3,3)
    wall_front=d.geom_xpos[wall]/1000-wr[:,2]*m.geom_size[wall,2]/1000
    probe=dict(socket_body='probe_socket',socket_parts=['Solid51','Solid36'],
        head_body='ball_and_probe',head_parts=['Solid38','Solid37'],
        spring_joint='probe_compression',spring_connection_bodies=['rigid_0011','probe_socket'],
        spring_endpoint_A=None,spring_endpoint_B=None,
        spring_endpoint_note='当前以关节弹簧表示轴向广义力，无两端attachment site/弹簧长度；不能把球心或body原点当成实物弹簧端点',
        available_connector_frames=[pmap[s]['native_frame'] for s in ['16763','16765','16766','16769','16759']],
        connector_physical_roles_confirmed=None,
        ball_center=site_record('ball_center_head'),socket_reference_center=site_record('ball_center_socket'),
        ball_radius_m=bg['ball']['radius_m'],ball_radius_provenance=src(MODEL/'data/ball_geometry.json','ball',STATUS_DERIVED),
        ball_joint='probe_passive_ball',physical_ball_angle_limits=None,
        working_face=dict(**probe_face,CAD_face_index=face['CAD_face_index'],
            CAD_local_center_m=face['center_native_m'],CAD_local_normal=face['normal_native'],
            radius_m=face['planar_face_radius_m'],outer_head_radius_m=face['head_outer_cylinder_radius_m'],
            normal_world=np.asarray(probe_face['world_pose']['rotation_matrix'])@np.array(face['normal_native']),
            lever_from_ball_world_m=np.asarray(probe_face['world_pose']['position_m'])-d.site('ball_center_head').xpos/1000,
            geometry_provenance=src(MODEL/'data/ball_geometry.json','probe_working_face',STATUS_DERIVED)),
        CAD_guide_axes=probe_axes,
        workpiece=dict(object='incline_surface',body='world',type='box',
            world_pose=pose(d.geom_xpos[wall],d.geom_xmat[wall]),half_sizes_m=m.geom_size[wall]/1000,
            front_face_point_world_m=wall_front,probe_approach_direction_world=wr[:,2],
            front_face_outward_normal_world=-wr[:,2],
            provenance=src(MODEL/'incline_press_model.py','build_incline_press:surface/rot',STATUS_DERIVED),
            contact_pair=dict(geom1='solid_16760',geom2='incline_surface',enabled=config['plate'],normal_only=True)),
        prescribed_point_contact=None,
        actual_probe_contacts=[c for c in contacts if 'solid_16760' in c['geoms']],
        contact_note='工作面site为几何中心和姿态标记，未绑定单一接触点；运行接触由整个探头mesh与试验板生成')

    missing=[
        dict(id='physical_anchors',objects=['A1@46','A2@46','A3@34','A4@34'],status=STATUS_MISSING,value=None,
             missing='实际绳孔/内壁固定与材料夹固的位置及出绳方向未获实物确认；模型点已经完整导出',
             impact='不阻塞当前仿真状态几何映射；阻碍把它认定为实物映射，A3/A4距34网格约0.0003m',
             recoverable='本工程可给点、最近表面和CAD候选，不能确认设计意图；需用户给具体附着特征/坐标'),
        dict(id='clamp_continuity',objects=['A3','A4','Solid34'],status=STATUS_MISSING,value=None,
             missing='橙紫之间真实夹固段形状、是否两处夹持、材料连续段及切向出绳方向',
             impact='当前模型用两个独立链端点表示，不能由共用电机推断实物中间连接；若局部两侧张力独立给定，不需中间无载段长度即可先算模型端点广义力',
             recoverable='源图和原生flex连接均已核对；实物夹固关系需用户/CAD指定'),
        dict(id='physical_route_lanes',objects=['L1','L2','L3','L4','L5','R2','R3','R5'],status=STATUS_MISSING,value=None,
             missing='实物Y方向车道、交叉处上下关系及全行程接触/脱离支路；当前±0.0005m车道和局部抬升是初始化假设',
             impact='给定本次节点状态可建立局部映射；仅给R和X_k两个标量不能唯一恢复含自由绳和横移/倾斜的当前模型走线',
             recoverable='可继续读取本工程任意已存qpos下的节点/接触；真实装配车道与绕行关系需用户确认'),
        dict(id='blue_L4_route_mismatch',objects=['蓝绳L3→A1末段','L4@rigid_0018'],status=STATUS_MISSING,value=None,
             missing='蓝绳设计不含L4，但当前有限直径初始化在L4有5个接触，最大重叠约0.000161212m；需确定实际是否接触及如何避让',
             impact='阻碍把名义蓝绳L3→A1无额外接触支路直接当作当前可行实物支路；参考节点/接触可读取但尚未平衡',
             recoverable='现有工程已可定位接触边号、点/法向/深度；真实蓝绳与L4的绕行关系需要用户确认，本轮不改线'),
        dict(id='ideal_slide_vs_current',objects=['rigid_0011','joint_16626','carriage_lateral_x','carriage_lateral_y','carriage_tilt_x','carriage_tilt_y'],
             status=STATUS_MISSING,value=None,missing='设计理想移动组与当前5自由度移动组的研究对象约定',
             impact='本次当前R、X_k坐标和其余自由度均已导出；二维构型几何映射须明确其余4个坐标',
             recoverable='本工程已提取差异；由研究者选择当前模型或设计理想约束，不在本轮修改'),
        dict(id='spring_attachment_points',objects=['rigid_0011','probe_socket','Solid41','Solid43','Solid44','Solid51','Solid36'],
             status=STATUS_MISSING,value=None,missing='实物弹簧两端承力连接点/端面、自由长度方向及连接件物理身份；当前只有关节弹簧',
             impact='不阻塞绳到R/X_k的几何映射；影响实物末端弹簧力线和偏心力矩定位',
             recoverable='所有相关CAD刚体坐标及杆/孔轴已提取；可从STEP继续筛选端面，但需用户确认哪两个面实际承簧'),
        dict(id='drum_detail',objects=['Sp1','Sp2','Sp3/Sp4'],status=STATUS_MISSING,value=None,
             missing='真实电机出绳孔、卷筒槽、初始材料相位、层径/绕线及卷筒固定端；现有为恒节圆半径+固定feed',
             impact='不阻塞局部张力到关节映射；后续电机收放几何需要补充',recoverable='节圆、轴、源hint和feed均可从工程获取且已导出；实物卷绕细节需用户'),
        dict(id='physical_limits',objects=['R','X_k','probe_passive_ball'],status=STATUS_MISSING,value=None,
             missing='实际机械行程及球窝摆角限位；当前这些joint limited=false',
             impact='不阻塞参考姿态局部映射；不能声称全行程有效',recoverable='CAD干涉可进一步分析，但不能代替用户给定机械限位'),
        dict(id='unique_contact_point',objects=['Solid37工作面','incline_surface'],status=STATUS_MISSING,value=None,
             missing='无预定义唯一点接触位置；已有面、法向和运行时contact',
             impact='绳到关节映射不依赖预选接触点；接触力矩分析要使用实际contact',recoverable='可读取后续状态生成的实际contact；本参考姿态无探头接触时不补造')]

    changed=[name for name,h in originals_before.items() if sha(ROOT/name)!=h]
    assert not changed, changed
    # Verify actual world positions against candidate source positions and material endpoints.
    errors=[]
    for r in routes:
        color=r['color']; c=cords[color]
        assert np.all(d.ten_length[c['tendon_ids']]>=0)
        err=abs(r['native_material_route']['total_tendon_length_m']-r['native_material_route']['sampled_polyline_length_m'])
        errors.append(err)
        assert err<1.e-12
        assert np.linalg.norm(r['native_material_route']['ordered_world_points_m'][-1]-next(a for a in anchors if a['color']==color)['world_pose']['position_m'])<1.e-12
    candidate_errors=[float(np.linalg.norm(d.site('route_'+n['sid']).xpos/1000-np.asarray(n['world_center_m'])))
                      for r in candidate['routes'] for n in r['nodes']]
    relevant_files={Path(getattr(mod,'__file__','')) for mod in list(sys.modules.values())
                    if getattr(mod,'__file__',None) and MODEL in Path(mod.__file__).parents}
    relevant_files.update({MODEL/'model_cable_drive_mm_g.xml',MODEL/'model_bend_assembly_mm_g.xml',
        MODEL/'data/corrected_routes_candidate.json',MODEL/'data/cad_manifest.json',MODEL/'data/ball_geometry.json',
        MODEL/'data/rigid_validation.json',MODEL/'data/carriage_alignment.json',MODEL/'data/solid5_update.json',
        ROOT/'four_cord_sim/experiment.json',ROOT/'four_cord_sim/sim.py',Path(__file__)})
    data=dict(meta=dict(project_root=str(ROOT),generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        scope='几何和连接核对；不辨识摩擦、弹性、迟滞，不设计估计器或控制器',
        python=platform.python_version(),python_executable=sys.executable,mujoco_version=mujoco.__version__,
        entry='four_cord_sim/sim.py run --config four_cord_sim/experiment.json',
        load_chain=['sim.py → whole_native_press.run → position_trial.run_position_trial',
            'incline_press_model.build_incline_press → model_cable_drive_mm_g.xml',
            'direct_carriage.release_carriage + carriage_contact.add_carriage_contact',
            'whole_guide_contact.add_whole_guides + whole_native_cords.add_native_cords',
            'mujoco.MjModel.from_xml_string → builder初始化 → mj_forward'],
        units=dict(output_length='m',output_angle='rad',orientation='quaternion wxyz / 3×3 local-to-parent/world matrix',
            raw_model='mm-g-s；MuJoCo数值无内置SI语义',length_to_SI=.001,angle_to_SI=1.,
            mesh='原装配CAD文件顶点m，mesh scale=1000转为模型mm；动态inline/代理依各自XML scale与代码原生mm，不再次统一缩放；编译后输出位置除1000',
            raw_xml_attributes='xml_attributes、xml、source_joint_mapping和raw_experiment_config保留原始单位；源附件保留mm/deg；其余输出几何统一m/rad'),
        statuses=[STATUS_XML,STATUS_RUNTIME,STATUS_DERIVED,STATUS_MISSING],
        compiler=dict(tree.find('compiler').attrib),includes=[dict(x.attrib) for x in tree.iter('include')],
        effective_model_sha256=sha(OUT/'effective_model.xml'),
        defaults=[ET.tostring(x,encoding='unicode') for x in tree.findall('default')],
        counts={k:int(getattr(m,k)) for k in ['nbody','njnt','nq','nv','nu','ntendon','nflex','neq','nkey']},
        original_models_unchanged=not changed,original_files_checked=len(originals_before),
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(relevant_files) if p.is_file()},
        note='model_new.xml为Simscape导入XML，不是MJCF；mujoco_migration及runs历史文件不是本次活动入口',
        unexecuted=['无mj_step/控制器运行','无新STEP重新网格化','不以历史仿真轨迹替代当前构型']),
        reference_state=dict(origin='当前experiment.json对应builder返回的初始化状态，随后mj_forward；不是keyframe，也不是纯qpos0',
            time_s=float(d.time),qpos_SI=qpos_si,qvel_SI=qvel_si,qpos0_SI=qpos0_si,
            coordinate_layout=layout,named_joint_qpos={j['name']:j['reference_qpos'] for j in joints},
            difference_from_qpos0='仅probe_compression由builder的初始预载平衡求根设置；其余沿qpos0；未积分',
            restore_instructions='加载effective_model.xml；将qpos_SI/qvel_SI按coordinate_layout中slide段乘1000，其余不变；赋给MjData，再mj_forward；XML本身不包含此初始化qpos',
            raw_experiment_config=config,initial_socket_retraction_m=float(d.joint('probe_compression').qpos[0])/1000,
            provenance=src(MODEL/'incline_press_model.py','build_incline_press:return model,data,spec',STATUS_RUNTIME)),
        physical_part_mapping=parts,body_tree=bodies,joints=joints,
        connections=dict(equality=[dict(x.attrib) for eq in tree.findall('equality') for x in eq],
            fixed_connection_rule='相同body的所有geom/site固连；无joint子body为父子固连；不需要额外weld equality',
            source_joint_connections=[j for j in source_graph['joint_connections'] if j['sid'] in ['16740','16626']],
            removed_upstream_joints=['joint_11549','joint_16729'],
            shared_axial_shaft='rigid_0017原紫轴去joint后作为rigid_0016子body，保留反向轴系；没有跨绳材料equality'),
        anchors=anchors,guides=guide_records,routes=routes,spools=spools,probe=probe,missing_geometry=missing,
        verification=dict(compile_and_forward=True,integration_steps=0,
            maximum_tendon_vs_polyline_length_error_m=max(errors),
            maximum_candidate_station_vs_runtime_error_m=max(candidate_errors),
            L5_owner=next(g['body'] for g in guide_records if g['label']=='L5'),
            material_chain_count=int(m.nflex),all_native_tendons_two_site=True,
            probe_contact_count=len(probe['actual_probe_contacts']),
            full_reference_contact_count=len(contacts),original_file_changes=changed))
    # Round-trip the exported artifact rather than trusting only in-memory builder data.
    restored=mujoco.MjModel.from_xml_path(str(OUT/'effective_model.xml'))
    rd=mujoco.MjData(restored)
    rd.qpos[:]=qpos_si;rd.qvel[:]=qvel_si
    for j in layout:
        if j['type']=='slide':
            rd.qpos[j['qpos_address']:j['qpos_address']+j['qpos_dimension']]*=1000
            rd.qvel[j['qvel_address']:j['qvel_address']+j['qvel_dimension']]*=1000
    mujoco.mj_forward(restored,rd)
    assert (restored.nq,restored.nv,restored.njnt,restored.ntendon)==(m.nq,m.nv,m.njnt,m.ntendon)
    site_error=float(np.max(abs(rd.site_xpos-d.site_xpos)))/1000
    length_error=float(np.max(abs(rd.ten_length-d.ten_length)))/1000
    assert site_error<1.e-12 and length_error<1.e-12 and rd.ncon==d.ncon
    mesh_error=max(p['mesh']['original_mesh_vs_compiled_bounds_error_m'] for p in parts
                   if p['mesh']['original_mesh_vs_compiled_bounds_error_m'] is not None)
    assert mesh_error<1.e-7
    assert all(p['mesh']['manifest_mesh_hash_matches'] for p in parts)
    assert all(p['source_step']['expected_sha256']==p['source_step']['current_sha256'] for p in parts)
    data['verification'].update(exported_xml_recompiled_and_SI_state_restored=True,
        roundtrip_maximum_site_error_m=site_error,roundtrip_maximum_tendon_length_error_m=length_error,
        maximum_raw_mesh_vs_compiled_native_bounds_error_m=mesh_error,
        all_relevant_mesh_and_STEP_hashes_match=True)
    write_json(ROOT/'geometry_data.json',data)
    write_report(data)
    write_excerpts(tree, source_graph)
    print(json.dumps(dict(status='OK',mujoco=mujoco.__version__,counts=data['meta']['counts'],
        output=['GEOMETRY_REPORT.md','geometry_data.json','extract_geometry.py','geometry_attachments/'],
        verification=data['verification']),ensure_ascii=False))


def vec(v):
    if v is None: return 'null'
    a=np.asarray(v)
    if a.ndim==0: return f'{float(a):.10g}'
    return '['+', '.join(f'{x:.10g}' for x in a.ravel())+']'


def write_report(a):
    lines=[]
    def add(s=''): lines.append(s)
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |');add('| '+' | '.join('---' for _ in headers)+' |')
        for row in rows: add('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |')
        add()
    add('# 本地 MuJoCo 机构几何提取报告')
    add()
    add('本轮仅提取几何、参考构型和连接关系。原工程未修改；没有积分运动、辨识绳参数、设计控制器或组装完整传力矩阵。未知值在 JSON 中使用 `null`。')
    add()
    add('## 1. 活动模型、单位与证据等级')
    add()
    add(f"工程根目录：`{ROOT}`。当前入口为 `four_cord_sim/sim.py run`，配置 `four_cord_sim/experiment.json`。实际加载 `four_cord_sim/model/model_cable_drive_mm_g.xml`，经 Python 动态增加末端压缩关节、试验板、共享轴、实体碰撞导向和四条材料绳；使用 `MjModel.from_xml_string` 编译。")
    add()
    add(f"实测 Python {a['meta']['python']}、MuJoCo {a['meta']['mujoco_version']}；解释器 `{a['meta']['python_executable']}`。成功编译并 `mj_forward`；未调用 `mj_step`。计数：{a['meta']['counts']}。根目录 `model_new.xml` 是 SimscapeMultibodyImportXML，不是 MJCF；`mujoco_migration` 与 `runs/*/model.xml` 为迁移或历史分支，本次没有拿它们代替当前模型。")
    add()
    add('加载/变更链：`sim.py → whole_native_press.run → position_trial.run_position_trial → incline_press_model.build_incline_press`；`direct_carriage.release_carriage` 释放横向/倾斜；`carriage_contact` 加碰撞代理；`whole_guide_contact.add_whole_guides` 加导向；`whole_native_cords.add_native_cords` 加有序分段 tendon 和 flex。完整有效 XML 附于 `geometry_attachments/effective_model.xml`，函数与关键 XML 摘录附于 `geometry_attachments/SOURCE_EXCERPTS.md`。')
    add()
    add('活动基准XML及生成XML没有 `<include>`、`<default>` 或 `<equality>`；没有keyframe。未显式写出的零位/关节pos/姿态等以编译后数组为准。上游推进 `joint_11549`、滚转 `joint_16729` 和导向辅助转动关节已在生成阶段移除；相关body保留固定变换。')
    add()
    add('统一输出 **m、rad**；方向向量无量纲，四元数 **wxyz**，旋转矩阵将局部向量变到父坐标/世界坐标。原生MJCF是 **mm–g–s**；不是SI。原CAD文件网格顶点为m，XML `scale="1000 1000 1000"` 变成模型mm；动态inline/碰撞代理按各自scale及代码使用原生mm，不能再统一乘1000。导出时编译后位置/尺寸除1000。编译器会重心化、旋转mesh，因此 `geom_pos/quat` 不能当作原CAD框架；应使用 `native_<SID>` site，JSON同时保留编译mesh变换。`xml_attributes/xml/source_joint_mapping/raw_experiment_config`及源码附件是保留原单位的溯源字段，和SI几何字段区分。')
    add()
    add('证据等级：**XML明确值**（含动态生成的XML）；**编译/运行时读取**（本次编译+forward）；**几何推算**（现有CAD解析、坐标变换、公切线/最近表面计算，均说明依据）；**缺失**（null）。模型内已定义不代表实物附着位置已经确认。既有CAD解析记录本轮未重新运行FreeCAD；相关STEP/网格及输入哈希保存在JSON，可核对来源。')
    add()
    add('## 2. 物理零件对应、刚体树与设计差异')
    add()
    table(['物理零件','SID / MuJoCo geom','所属body','确认依据'],[
        [p['physical_part'],(p['source_sid'] if p['actual_source_sid']==p['source_sid'] else p['actual_source_sid']+'（兼容别名'+p['source_sid']+'）')+' / '+p['geom'],p['body'],'源图名+当前XML/native site；编译确认固连'] for p in a['physical_part_mapping']])
    add('Solid34、Solid42、Solid43、Solid41同属 `rigid_0011`，且还包含Solid35/40/44。51+36属于 `probe_socket`；38+37属于 `ball_and_probe`。模型中的 `solid_16768/native_16768` 已对应Solid5（源SID17637），不能继续按旧Solid47解释。')
    add()
    add('**设计与实现差异：** 设计M只沿46轴向平移且不相对转动；当前主配置为 `direct_contact`，M有轴向移动、两个横移、两个倾斜，共5个DOF，轴向自转仍受约束。模型坐标映射必须携带这些附加坐标，不能仅由(R,X_k)唯一确定全部末端/绳路。参考姿态这4个附加坐标为0，但不是全行程约束。')
    add()
    add('**导向归属：** L1、L2、L3、L4均固定于 `rigid_0018`（Solid5上游支承），不随R或伸缩运动；L5固定于 `rigid_0008`（Solid46），随R、不随X_k。R5/R2/R3也固定在上游支承。本次活动分支已锁定轮/辅助环的独立转动；不能从P编号推断独立实物轮。')
    add()
    add('父子树（必要祖先及固定辅助body全部保留；材料绳粒子按routes单独展开）：')
    add()
    add('```text')
    bodies={b['name']:b for b in a['body_tree']}
    def depth(b):
        n=0
        while b['parent'] is not None:
            n+=1;b=bodies[b['parent']]
        return n
    for b in a['body_tree']:
        add('  '*depth(b)+str(b['name'])+' : '+(', '.join(str(j) for j in b['joints']) if b['joints'] else b['connection']))
    add('```');add()
    add('每个body的父系局部平移与旋转（编译值；m、wxyz）：');add()
    table(['body → parent','局部pos m','局部quat wxyz'],[
        [b['name']+' → '+str(b['parent']),vec(b['local_position_m']),vec(b['local_quaternion_wxyz'])] for b in a['body_tree']])
    add('同body内的geom/site为固连；无joint子body为父子固连；没有额外weld equality。橙紫共享轴通过重挂紫卷绕环为橙轴的无joint子body实现，不是两条绳的材料连接约束。')
    add()
    add('## 3. 关节与坐标地址')
    add()
    add('所有地址从0开始。轴点局部属于joint所属body；世界轴点是本参考状态下 `data.xanchor`。slide的该点是轴线标记，不是实物弹簧端点；ball没有唯一转轴，方向为null。`ref`几何零位与`springref`弹簧平衡参考分开导出。未启用限位的`range`是null，不能把XML未生效的上下界或算法测试域当作机械限位。')
    add()
    table(['joint / 用途','body → parent','类型','局部轴点m / 方向','世界轴点m / 方向','qpos地址×维 / qvel地址×维','有效range'],[
        [j['name']+(' / R' if j['name']==BEND else ' / X_k' if j['name']==SLIDE else ''),
         j['body']+' → '+j['parent'],j['type'],vec(j['local_axis_point_m'])+' / '+vec(j['local_axis_direction']),
         vec(j['world_axis_point_m'])+' / '+vec(j['world_axis_direction']),
         f"{j['qpos_address']}×{j['qpos_dimension']} / {j['qvel_address']}×{j['qvel_dimension']}",vec(j['range'])] for j in a['joints']])
    add('R正轴约world −Y；X_k正轴约world +Z，正值回缩，伸出为−q。当前 `probe_compression` 正值同样回缩。球铰为4维单位四元数/3维角速度，不可将其qpos视为3个欧拉角。两个倾斜关节轴按body内顺序串接，非零姿态时应重新forward读取轴。')
    add()
    add('## 4. 固定点、材料夹固点和其他端点')
    add()
    table(['点 / site','所属body / 物理零件','局部body m','物理CAD框架 m','世界 m','用途/确认'],[
        [x['label']+' / '+x['object'],x['body']+' / '+x['physical_part'],vec(x['local_position_m']),
         vec(x['position_in_physical_CAD_frame_m']),vec(x['world_pose']['position_m']),
         x['role']+'；运行读取，实物特征待确认'] for x in a['anchors']])
    add('A1/A2是46上的固定绳端；A3/A4是34上两个模型端点，表征材料夹固的两侧。当前橙、紫由**两条独立flex链**实现，并没有一条连续flex穿过34、没有34上的中间材料段或夹持器几何。共同电机只同步两侧相反收放；不能据此认定中间连续绳已显式建模。')
    add()
    add('当前原始网格最近表面检查（几何推算，不确认具体孔/内壁）：');add()
    table(['点','最近距离m','最近表面世界点m'],[[x['label'],vec(x['surface_check']['distance_m']),vec(x['surface_check']['nearest_point_world_m'])] for x in a['anchors']])
    table(['点','指向上一材料节点的世界单位向量','相同方向在所属body中的单位向量'],[[x['label'],vec(x['toward_previous_material_point_world_unit']),vec(x['toward_previous_material_point_body_unit'])] for x in a['anchors']])
    add('A3/A4约0.0003m的表面距离是现有模型的偏离，不在本轮吸附/修正坐标。A1靠近表面也不足以认定它正是设计中的46内壁孔。所有feed固定端、每个自由粒子site和末端锚点的顺序与归属详见JSON routes；feed在world上，不是随电机旋转的实体端点。')
    add()
    add('## 5. 导向尺寸、轴线与姿态')
    add()
    add('以下center/axis是实际物理CAD导向的中心与轴，非每条绳的独立P接触框架；局部值属于所列body。CAD有限圆柱由解析表面/包围尺寸读取；槽轮由CAD同轴圆剖面恢复。姿态以同名 `native_CAD_frame` 的wxyz/matrix及各编译碰撞geom记录在JSON；圆柱绕自身轴角不影响接触几何。');add()
    table(['导向 / body','局部中心m / 轴','世界中心m / 轴','实体半径m / 轴向全长m','名义节圆半径m','类型/来源'],[
        [g['label']+' / '+g['body'],vec(g['center_local_body_m'])+' / '+vec(g['axis_local_body']),
         vec(g['center_world_m'])+' / '+vec(g['axis_world']),
         vec(g['physical_radius_m'])+' / '+vec(g['full_axial_length_m']),vec(g['nominal_pitch_radius_m']),g['type']+'；whole_guide_contact+CAD记录+forward'] for g in a['guides']])
    add('槽轮(R5/R2/R3)局部轴向位置—外半径剖面（m，轴向零点为槽底平面）：`'+str((np.array(wheel_profile()[0])/1000).tolist())+'`；孔半径 `'+vec(wheel_profile()[1]/1000)+'` m。实体槽底半径0.00475m与名义节圆0.00476m不同；L1–L3实体半径0.00075m与节圆0.00076m不同。原生绳半径为0.0002m；初始化按实体表面+绳半径、槽斜率以及车道计算中心线，不能将节圆候选切点直接当当前有限直径绳切点。')
    add()
    add('## 6. 四色有序路径、支路与参考长度')
    add()
    add('途经不等于接触。`route_<SID>`是源路径/几何框架，不会自动强制绳通过；当前每个native spatial tendon只有相邻两个site，没有`geom`绕行项。接触由flex边与`wc_guide_*`碰撞面产生。JSON逐一保留名义站序、支路符号、公切线与圆弧切点，及当前原生材料链所有site/tendon的完整顺序与长度。');add()
    for r in a['routes']:
        add('### '+r['color']);add()
        add('名义顺序：'+' → '.join(s['object']+'('+label+')' for s,label in zip(r['ordered_station_sites'],r['ordered_station_labels']))+'。');add()
        if r['color']=='blue': add('设计简写是电机→L1→L2→L3→A1；模型在前端之前另有R2。蓝在L1/L2的world +X侧通过，名义候选中两者有间隙；L3保留+X侧出口。')
        elif r['color']=='green': add('设计简写是电机→L1→L2→A2；模型在之前另有R3，L1/L2保留S形相反曲率。')
        elif r['color']=='orange': add('模型完整上游含R5→R2，随后在O/L4的world +X侧通过到A3；R2取向R3的内侧（world −X）。名义候选允许R2/L4脱离，不强造接触。')
        else: add('模型含上游R3，并在L1/L2/L3的world +X侧通过，另有L4的world −X侧途经/接触，然后L5回转到A4。L4是橙紫共用的同一个实体，不能漏掉或新增第二个轮。L5为短支路回转，不是任意整圈绕线。')
        add()
        table(['站点 / 碰撞实体','名义接触','名义XZ有符号包角rad','名义绕侧'],[
            ['route_'+g['sid']+' / '+g['physical_label'],str(g['active']),vec(g['signed_angle_XZ_rad']),g.get('mean_radial_world_X_side') or 'null（无接触）'] for g in r['nominal_route']['guides']])
        n=r['native_material_route']
        add(f"当前材料链：`{n['ordered_sites'][0]}` → {len(n['ordered_sites'])-2}个自由粒子site → `{n['anchor_site']}`；flex `{n['flex']}`，{len(n['ordered_tendons'])}段tendon。名义候选弧/直段合长 **{r['nominal_route']['length_m']:.12g} m**；本次forward原生tendon合长 **{n['total_tendon_length_m']:.12g} m**。当前生成guide接触的实体为 {', '.join(n['current_guide_contact_labels']) or '无'}，接触点/法向见附件索引；接触不必有正载荷。")
        add()
        table(['当前接触顺序（按材料边号）','导向body','边号区间（从0）','最小有符号间隙m','激活接触约束数'],[
            [g['label'],g['body'],f"{g['first_material_edge']}–{g['last_material_edge']}",vec(g['minimum_signed_gap_m']),g['constraint_active_contact_count']]
            for g in n['current_contact_order_along_material_edges']])
        if r['color']=='blue':
            add('**设计不一致：蓝绳当前在L4还有5个激活接触，最深重叠约0.000161212m，位于L3→A1段；设计路径与名义候选没有蓝绳L4接触。** 本次状态只是未积分的初始化，不把穿叠解释成稳定贴绕；本轮保留原模型及这个缺口。');add()
    add('名义切点位于JSON `routes[].nominal_route.spans[].from_world_m/to_world_m` 与 `guides[].entry_world_m/exit_world_m`，为参考零位候选的几何推算，未接触项为null；本次状态中绳自由节点形成分段线，当前唯一光滑切点不可定义，记null，提供全部节点和MuJoCo实际contact作为替代证据。名义切点不能被标成当前原生接触切点。');add()
    add('初始Y车道（蓝+0.0005m、绿−0.0005m），以及橙、紫在绿末端引线交叉处的自由Y抬升，是 `whole_native_cords.py` 明写的初始化假设，没有创建固定途经点。源候选文件也明确锚点、轴向车道未经过实物确认。所有切点、长度、接触支路均依赖姿态和绳节点状态，不能从本表推断全行程恒定力臂。');add()
    add('卷筒/电机接口：');add()
    table(['颜色','源station / 当前电机joint','节圆半径m','原关节每rad放绳m','实际固定feed'],[
        [s['color'],s['source_station']+' / '+s['current_motor_joint'],vec(s['pitch_radius_m']),vec(s['payout_m_per_original_joint_rad']),s['actual_native_feed_site']] for s in a['spools']])
    table(['颜色','源节圆中心世界m','原源节圆轴方向world','实际feed世界m'],[
        [s['color'],vec(s['station']['world_pose']['position_m']),vec(s['source_spool_axis_world']),
         vec(next(r['native_material_route']['feed']['world_pose']['position_m'] for r in a['routes'] if r['color']==s['color']))] for s in a['spools']])
    add('放绳符号列针对各自原源关节，紫原joint_16739在当前共享轴中移除，不可直接当共享电机符号。当前橙紫由 `set_native_feed` 做相反收放，电机正向收紫放橙；蓝绿独立。真实卷筒槽、材料固定端和多层卷绕未实现，不阻塞局部张力输入。');add()
    add('## 7. 明确参考姿态与坐标')
    add()
    add('本次使用活动builder返回的初始化状态、t=0，再执行forward。该状态包含builder求解的微小压缩止挡平衡位移，**不等于纯qpos0**；不是keyframe。全量qpos/qvel及qpos0已经SI换算，完整地址表在 `reference_state.coordinate_layout`；9020个joint中的绳粒子匿名joint其name保留null，以id/body识别，不编造XML名称。重新加载XML后，必须将JSON中各slide坐标/速度乘1000还原原生mm，其余维度原样赋值，再forward；XML本身不保存这份初始化qpos。');add()
    table(['joint','qpos0','参考qpos','单位','弹簧参考'],[
        [j['name'],vec(j['qpos0']),vec(j['reference_qpos']),j['qpos_unit'],vec(j['spring_reference'])] for j in a['joints']])
    add('R、X_k、所有锚点、导向中心/轴已在上表分别列局部/世界值；完整父系矩阵与物理CAD框架见JSON。绳粒子的qpos=0表示其相对初装body平移为0，不表示该粒子的世界位置为原点。');add()
    add('## 8. 末端回缩、球头、探头及工件')
    add()
    p=a['probe'];f=p['working_face'];w=p['workpiece']
    add('`rigid_0011 → probe_socket` 由 `probe_compression` 轴向滑动连接；`probe_socket → ball_and_probe` 由 `probe_passive_ball` 连接。51/36固连为回缩座；38/37固连为球头探头。原始源刚体曾把整组焊在一起，当前MJCF已明确拆分，不沿用旧源图替代当前运动关系。');add()
    add('当前弹簧是 `probe_compression` 的广义坐标弹簧；有效范围 [0,0.0008]m，参考状态有微小负止挡变形。两端实物连接点/端面 **null**，模型没有两个弹簧attachment site，不能把关节原点或球心补当弹簧端点。Solid41/43/44/51/36的完整CAD框架已导出，但哪两个面实际承簧仍未知。');add()
    table(['几何','局部定义 / 所属体','参考世界值','来源/状态'],[
        ['球头中心',vec(p['ball_center']['local_position_m'])+' / ball_and_probe',vec(p['ball_center']['world_pose']['position_m']),'ball_center_head；运行读取'],
        ['球面半径',vec(p['ball_radius_m'])+' m / Solid38','—','CAD两个球面解析；几何推算'],
        ['探头工作面中心',vec(f['CAD_local_center_m'])+' m / Solid37 CAD',vec(f['world_pose']['position_m']),'Solid37 STEP Face2+probe_working_face；几何推算/运行读取'],
        ['工作面法向',vec(f['CAD_local_normal'])+' / Solid37 CAD',vec(f['normal_world']),'CAD定义+姿态变换；几何推算'],
        ['工作面姿态wxyz','相对ball_and_probe '+vec(f['local_quaternion_wxyz']),vec(f['world_pose']['quaternion_wxyz']),'probe_working_face；运行读取'],
        ['工作面半径',vec(f['radius_m'])+' m / Solid37','—','CAD外边界圆；几何推算'],
        ['球心到工作面中心','ball_and_probe中 '+vec(f['local_position_m'])+' m',vec(f['lever_from_ball_world_m']),'CAD+运行姿态'],
        ['试验板前表面一点','world固定',vec(w['front_face_point_world_m']),'incline_press_model，按倾角/间隙构造；几何推算'],
        ['试验板朝探头外法向','world',vec(w['front_face_outward_normal_world']),'box局部−Z面法向；几何推算']])
    add('探头外圆半径0.002m，前端工作圆面半径0.0018m，两者不可混用。板是有限box，半尺寸m '+vec(w['half_sizes_m'])+'，中心世界m '+vec(w['world_pose']['position_m'])+'，姿态wxyz '+vec(w['world_pose']['quaternion_wxyz'])+'。这是已存在试验工件，不是实物工件标定。碰撞对 `solid_16760`（整个探头mesh）—`incline_surface`，法向接触；没有指定单一点接触site。');add()
    add(f"本参考姿态探头实际contact数量：{len(p['actual_probe_contacts'])}；唯一接触点为null。工作面法向与contact法向分别导出，不把工作面几何中心默认成接触点。全模型{a['verification']['full_reference_contact_count']}个生成contact（含绳/导向、绳间及实体代理）在 `reference_contacts.json`，附对象顺序和法向；未据此估计张力。")
    add()
    table(['已有杆/孔轴','所属body / CAD局部点与方向','参考世界点与方向','半径m / 来源'],[
        [g['label'],g['body']+' / '+vec(g['local_CAD_point_m'])+' / '+vec(g['local_CAD_direction']),
         vec(g['world_point_m'])+' / '+vec(g['world_direction']),vec(g['radius_m'])+' / ball_geometry CAD surface'] for g in p['CAD_guide_axes']])
    add('上述杆/孔的实际偏心信息保留在既有CAD记录；压缩joint的作用线仍以本次joint轴表为准，不能把任一CAD轴静默替换为关节轴。');add()
    add('## 9. 缺口、可提取边界与验收')
    add()
    table(['缺项','具体点/轴/关系','影响','本工程/用户边界'],[[x['id'],x['missing'],x['impact'],x['recoverable']] for x in a['missing_geometry']])
    add('对给定构型和接触支路，R/X_k的轴、锚点、移动组/46/L5归属、所有材料段方向与导向接触点已可供另一位助手组装局部张力的几何合力/力矩。不能只累加绳端力而漏掉固定于46的L5反力；当前额外横移/倾斜及末端被动坐标应作为给定状态。原生模型含独立运动的绳节点，几何导出不意味着其含惯性瞬态的全部接触反力可仅由张力唯一恢复。若采用质量可忽略的理想局部传力，应由后续建模者明确该约定。当前蓝绳/L4初始穿叠尚未解决；若研究对象是可行的真实走线或设计理想移动副，须确认该支路及锚点/夹固/车道，不能将初始化接触表当作已经验证的实际走线。');add()
    add('没有把缺少摩擦、弹性或迟滞当作本轮几何失败。本次也不根据同一根绳假设各段张力相等。只有几何方向/作用位置参与后续映射；给定局部张力由后续研究者提供。');add()
    v=a['verification']
    add(f"校核：编译和forward通过；四flex、{a['meta']['counts']['ntendon']}段原生tendon；合长与节点折线最大误差 {v['maximum_tendon_vs_polyline_length_error_m']:.3g}m；候选station与当前零位site最大误差 {v['maximum_candidate_station_vs_runtime_error_m']:.3g}m；每条末端site与锚点一致；L5归属rigid_0008；检查{a['meta']['original_files_checked']}份原工程代码/XML/JSON，内容哈希未变。未做长仿真、全行程验证、新STEP转换或实物附着确认。")
    add()
    add(f"导出复载验证：有效XML重新编译，按SI地址表还原qpos/qvel后，全部site最大差 {v['roundtrip_maximum_site_error_m']:.3g}m，所有tendon长度最大差 {v['roundtrip_maximum_tendon_length_error_m']:.3g}m，contact数量一致。21项相关CAD的STEP/网格哈希均匹配；原文件顶点与编译网格逆变换回CAD坐标后的边界最大差 {v['maximum_raw_mesh_vs_compiled_native_bounds_error_m']:.3g}m，确认没有额外1000倍缩放。CAD解析边界与离散网格的差值另外保留，不把两者混为一谈。")
    add()
    add('复现：在工程根目录运行 `four_cord_sim/.venv/bin/python -B extract_geometry.py`。该脚本只写本报告、JSON与附件，完整输入哈希在meta。JSON中的raw/XML字符串保留原始单位，其余几何字段均为m/rad。')
    add()
    add('交付文件：`GEOMETRY_REPORT.md`、`geometry_data.json`、`extract_geometry.py`、`geometry_attachments/effective_model.xml`、`geometry_attachments/reference_contacts.json`、`geometry_attachments/SOURCE_EXCERPTS.md`。')
    (ROOT/'GEOMETRY_REPORT.md').write_text('\n'.join(lines)+'\n')


def write_excerpts(tree,graph):
    lines=['# 关键来源附件','', '本附件中的XML/代码按源文件保留原始单位；不是统一SI数值表。', '']
    for file, functions in [('incline_press_model.py',['make_shared_axial_shaft','build_incline_press']),
                            ('direct_carriage.py',['release_carriage']),
                            ('whole_guide_contact.py',['guide_specs','add_whole_guides']),
                            ('whole_native_cords.py',['_initial_polyline','_four_paths','add_native_cords','set_native_feed'])]:
        module=sys.modules[Path(file).stem]
        for func in functions:
            obj=getattr(module,func);code,start=inspect.getsourcelines(obj)
            lines.extend([f'## four_cord_sim/model/{file}:{start} — {func}','','```python',''.join(code).rstrip(),'```',''])
    source=ET.parse(MODEL/'model_cable_drive_mm_g.xml').getroot()
    for name in ['rigid_0008','rigid_0011','probe_socket','ball_and_probe']:
        body=source.find(f".//body[@name='{name}']")
        small=ET.Element('body',body.attrib)
        for child in body:
            if child.tag in ('joint','site','geom'):
                small.append(ET.fromstring(ET.tostring(child)))
        lines.extend([f'## 原XML model_cable_drive_mm_g.xml — body {name}','','```xml',ET.tostring(small,encoding='unicode'),'```',''])
    lines.extend(['## 当前生成XML：joint与末端contact','','```xml'])
    for j in tree.iter('joint'):
        if j.get('name'):lines.append(ET.tostring(j,encoding='unicode').strip())
    for c in tree.findall('contact'): lines.append(ET.tostring(c,encoding='unicode').strip())
    lines.extend(['```','', '## 源连接图说明','',
                  '`mujoco_migration/data/source_graph.json` 保留源块SID/连线；当前几何以有效MJCF为准。源图Solid5使用兼容别名SID16768，实际SID17637。源刚体把末端焊在一起，当前模型已拆为socket/head，报告没有将源刚体关系当成当前连接。'])
    (OUT/'SOURCE_EXCERPTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    extract()
