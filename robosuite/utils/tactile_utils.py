"""
Utility functions for generating tactile sensor arrays in MuJoCo XML.

Provides functions to create 2D grids of MuJoCo touch sensors on gripper fingerpads.
The generated sensor arrays produce outputs analogous to single-channel images (H, W, 1).

This module is gripper-agnostic: it generates site and sensor XML elements that can be
injected into any gripper XML model at runtime.
"""

import xml.etree.ElementTree as ET
import numpy as np


def generate_tactile_grid(
    parent_body_element,
    sensor_element,
    pad_center,
    pad_halfsize,
    normal_axis,
    prefix,
    rows=8,
    cols=8,
    site_group="1",
):
    """
    Generate a 2D grid of MuJoCo touch sensor sites and corresponding <touch> sensor
    elements, then inject them into the provided XML elements.

    Each touch sensor is bound to a small <site> placed on the fingerpad surface.
    The sites are arranged in a regular grid covering the pad area. The output of all
    sensors for one pad can be reshaped into (rows, cols, 1), analogous to a
    single-channel image.

    Args:
        parent_body_element (ET.Element): The <body> XML element (e.g., finger_joint1_tip)
            into which the <site> elements will be appended.
        sensor_element (ET.Element): The <sensor> XML element into which the <touch>
            elements will be appended.
        pad_center (array-like): (x, y, z) center of the fingerpad in the parent body frame.
        pad_halfsize (array-like): (hx, hy, hz) half-sizes of the pad box geom.
            The tactile grid spans the face perpendicular to @normal_axis.
        normal_axis (str): One of "+x", "-x", "+y", "-y", "+z", "-z". Indicates
            which face of the pad box the sensors are placed on (contact surface normal).
        prefix (str): Name prefix for sensors and sites, e.g. "left_pad".
            Sensor names will be "{prefix}_touch_{row}_{col}".
        rows (int): Number of rows in the tactile grid.
        cols (int): Number of columns in the tactile grid.
        site_group (str): MuJoCo site group for visualization control.

    Returns:
        list of str: List of sensor names (un-prefixed by gripper naming_prefix)
            in row-major order, length = rows * cols.
    """
    pad_center = np.array(pad_center, dtype=np.float64)
    pad_halfsize = np.array(pad_halfsize, dtype=np.float64)

    # Determine which two axes form the grid plane and which is the normal
    axis_map = {"x": 0, "y": 1, "z": 2}
    normal_sign = 1.0 if normal_axis[0] == "+" else -1.0
    normal_idx = axis_map[normal_axis[1]]

    # The two tangent axes (grid plane)
    tangent_axes = [i for i in range(3) if i != normal_idx]
    ax_u, ax_v = tangent_axes  # u -> cols, v -> rows (or vice versa, we define row=v, col=u)

    # Site dimensions: each site covers one cell of the grid
    cell_halfsize_u = pad_halfsize[ax_u] / cols
    cell_halfsize_v = pad_halfsize[ax_v] / rows
    # Along normal direction, site covers full pad depth (co-located with collision sub-geom)
    site_halfsize_normal = pad_halfsize[normal_idx]

    # Build site size array
    site_size = np.zeros(3)
    site_size[ax_u] = cell_halfsize_u
    site_size[ax_v] = cell_halfsize_v
    site_size[normal_idx] = site_halfsize_normal

    sensor_names = []

    for r in range(rows):
        for c in range(cols):
            # Compute position of this cell center
            # Linearly space from -halfsize + cell_half to +halfsize - cell_half
            u_pos = -pad_halfsize[ax_u] + cell_halfsize_u * (2 * c + 1)
            v_pos = -pad_halfsize[ax_v] + cell_halfsize_v * (2 * r + 1)

            pos = pad_center.copy()
            pos[ax_u] += u_pos
            pos[ax_v] += v_pos
            # pos along normal stays at pad_center (co-located with subdivision geom)

            site_name = f"{prefix}_touch_site_{r}_{c}"
            sensor_name = f"{prefix}_touch_{r}_{c}"

            # Create site element
            site_elem = ET.SubElement(parent_body_element, "site")
            site_elem.set("name", site_name)
            site_elem.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            site_elem.set("size", f"{site_size[0]:.6f} {site_size[1]:.6f} {site_size[2]:.6f}")
            site_elem.set("type", "box")
            site_elem.set("group", site_group)
            site_elem.set("rgba", "0.3 0.8 0.3 0.3")  # semi-transparent green for visualization

            # Create touch sensor element
            touch_elem = ET.SubElement(sensor_element, "touch")
            touch_elem.set("name", sensor_name)
            touch_elem.set("site", site_name)

            sensor_names.append(sensor_name)

    return sensor_names


def subdivide_pad_collision(
    parent_body_element,
    pad_center,
    pad_halfsize,
    normal_axis,
    prefix,
    rows=8,
    cols=8,
    geom_attribs=None,
):
    """
    Replace a single pad collision geom with a grid of smaller collision geoms.

    A single large collision box generates only a few contact points at the
    edges of the contact patch. By subdividing into many small boxes, each cell
    independently generates contacts with the grasped object, producing a
    spatially distributed force map that the co-located touch sensor sites can
    capture.

    Args:
        parent_body_element (ET.Element): The <body> element to append geoms to.
        pad_center (array-like): (x, y, z) center of the original pad geom.
        pad_halfsize (array-like): (hx, hy, hz) half-sizes of the original pad geom.
        normal_axis (str): "+x"/"-x"/"+y"/"-y"/"+z"/"-z" contact surface normal.
        prefix (str): Name prefix, e.g. "left_pad".
        rows (int): Grid rows.
        cols (int): Grid columns.
        geom_attribs (dict or None): Attributes copied from the original geom
            (friction, solref, conaffinity, contype, etc.).

    Returns:
        list of str: Un-prefixed geom names in row-major order.
    """
    pad_center = np.array(pad_center, dtype=np.float64)
    pad_halfsize = np.array(pad_halfsize, dtype=np.float64)

    axis_map = {"x": 0, "y": 1, "z": 2}
    normal_idx = axis_map[normal_axis[1]]
    tangent_axes = [i for i in range(3) if i != normal_idx]
    ax_u, ax_v = tangent_axes

    cell_halfsize_u = pad_halfsize[ax_u] / cols
    cell_halfsize_v = pad_halfsize[ax_v] / rows

    geom_size = np.zeros(3)
    geom_size[ax_u] = cell_halfsize_u
    geom_size[ax_v] = cell_halfsize_v
    geom_size[normal_idx] = pad_halfsize[normal_idx]  # full depth

    if geom_attribs is None:
        geom_attribs = {}

    # Attributes managed explicitly; everything else is inherited from original geom
    _SKIP_ATTRS = {"name", "type", "pos", "size", "group"}

    geom_names = []

    for r in range(rows):
        for c in range(cols):
            u_pos = -pad_halfsize[ax_u] + cell_halfsize_u * (2 * c + 1)
            v_pos = -pad_halfsize[ax_v] + cell_halfsize_v * (2 * r + 1)

            pos = pad_center.copy()
            pos[ax_u] += u_pos
            pos[ax_v] += v_pos
            # normal position stays at pad_center (same depth as original geom)

            geom_name = f"{prefix}_pad_geom_{r}_{c}"

            geom_elem = ET.SubElement(parent_body_element, "geom")
            geom_elem.set("name", geom_name)
            geom_elem.set("type", "box")
            geom_elem.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
            geom_elem.set("size", f"{geom_size[0]:.6f} {geom_size[1]:.6f} {geom_size[2]:.6f}")
            geom_elem.set("group", "0")  # collision group

            # Copy contact / solver properties from original geom
            for attr_name, attr_val in geom_attribs.items():
                if attr_name not in _SKIP_ATTRS:
                    geom_elem.set(attr_name, attr_val)

            geom_names.append(geom_name)

    return geom_names


def patch_model_xml_with_tactile(xml_string, gripper_prefix="gripper0_", rows=8, cols=8):
    """
    Patch a stored model XML string to replace standard gripper pad collision
    geoms with subdivided collision geoms and add tactile sensor sites / sensors.

    This is needed when replaying datasets recorded with a standard gripper but
    processing them with a tactile-gripper environment.  The stored XML contains
    the original single ``finger*_pad_collision`` geoms; this function replaces
    them with the same subdivision grid that ``PandaTactileGripper`` creates at
    runtime, so that ``generate_id_mappings`` can resolve every geom name that
    the gripper model object references.

    Args:
        xml_string (str): The full model XML string (as stored in the dataset).
        gripper_prefix (str): Naming prefix for the gripper (default ``"gripper0_"``).
        rows (int): Number of rows in the tactile grid.
        cols (int): Number of columns in the tactile grid.

    Returns:
        str: The patched XML string.  If no matching pad geoms are found the
        original string is returned unchanged.
    """
    root = ET.fromstring(xml_string)

    # Ensure a <sensor> element exists under root
    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")

    pad_configs = [
        dict(
            body_name=f"{gripper_prefix}finger_joint1_tip",
            geom_name=f"{gripper_prefix}finger1_pad_collision",
            pad_center=[0.0, -0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="-y",
            prefix=f"{gripper_prefix}left_pad",
        ),
        dict(
            body_name=f"{gripper_prefix}finger_joint2_tip",
            geom_name=f"{gripper_prefix}finger2_pad_collision",
            pad_center=[0.0, 0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="+y",
            prefix=f"{gripper_prefix}right_pad",
        ),
    ]

    patched = False
    for cfg in pad_configs:
        # Locate the fingertip <body>
        tip_body = None
        for body in root.iter("body"):
            if body.get("name") == cfg["body_name"]:
                tip_body = body
                break
        if tip_body is None:
            continue

        # Find the original single pad collision geom
        pad_geom = None
        for child in list(tip_body):
            if child.tag == "geom" and child.get("name") == cfg["geom_name"]:
                pad_geom = child
                break
        if pad_geom is None:
            # Already patched or different gripper structure – skip
            continue

        geom_attribs = dict(pad_geom.attrib)
        tip_body.remove(pad_geom)

        # Add grid of small collision geoms (names already fully prefixed)
        subdivide_pad_collision(
            parent_body_element=tip_body,
            pad_center=cfg["pad_center"],
            pad_halfsize=cfg["pad_halfsize"],
            normal_axis=cfg["normal_axis"],
            prefix=cfg["prefix"],
            rows=rows,
            cols=cols,
            geom_attribs=geom_attribs,
        )

        # Add co-located sensor sites + <touch> elements
        generate_tactile_grid(
            parent_body_element=tip_body,
            sensor_element=sensor_elem,
            pad_center=cfg["pad_center"],
            pad_halfsize=cfg["pad_halfsize"],
            normal_axis=cfg["normal_axis"],
            prefix=cfg["prefix"],
            rows=rows,
            cols=cols,
        )
        patched = True

    if patched:
        return ET.tostring(root, encoding="unicode")
    return xml_string


def reshape_tactile_reading(sensor_values, rows=8, cols=8):
    """
    Reshape a flat array of tactile sensor readings into a 2D image-like array.

    Args:
        sensor_values (np.array): Flat array of shape (rows * cols,) containing
            touch sensor readings.
        rows (int): Number of rows in the tactile grid.
        cols (int): Number of columns in the tactile grid.

    Returns:
        np.array: Array of shape (rows, cols, 1), analogous to a single-channel image.
    """
    assert len(sensor_values) == rows * cols, \
        f"Expected {rows * cols} sensor values, got {len(sensor_values)}"
    return sensor_values.reshape(rows, cols, 1)
