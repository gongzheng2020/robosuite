"""
Gripper for Franka's Panda with tactile sensor arrays on fingerpads.

Extends the standard PandaGripper by dynamically injecting 2D grids of MuJoCo
touch sensors onto both fingerpads. The tactile output for each pad is shaped
as (rows, cols, 1), analogous to a single-channel image.
"""
import numpy as np

from robosuite.models.grippers.panda_gripper import PandaGripperBase, PandaGripper
from robosuite.utils.mjcf_utils import xml_path_completion, find_elements
from robosuite.utils.tactile_utils import generate_tactile_grid, subdivide_pad_collision


# Default tactile grid resolution
DEFAULT_TACTILE_ROWS = 8
DEFAULT_TACTILE_COLS = 8


class PandaTactileGripperBase(PandaGripperBase):
    """
    Panda gripper with tactile sensor arrays on both fingerpads.

    The touch sensors are dynamically generated and injected into the XML after
    loading, so the base XML file (panda_tactile_gripper.xml) remains clean and
    identical to the standard panda_gripper.xml.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
        tactile_rows (int): Number of rows in each tactile sensor grid
        tactile_cols (int): Number of columns in each tactile sensor grid
    """

    def __init__(self, idn=0, tactile_rows=DEFAULT_TACTILE_ROWS, tactile_cols=DEFAULT_TACTILE_COLS):
        self._tactile_rows = tactile_rows
        self._tactile_cols = tactile_cols
        self._tactile_sensor_names = {"left": [], "right": []}
        self._left_pad_geom_names = []
        self._right_pad_geom_names = []

        # Load the base XML (identical to panda_gripper.xml)
        # NOTE: super().__init__ calls GripperModel.__init__ which calls MujocoXMLModel.__init__
        # which parses elements and applies naming prefix. We need to inject sensors BEFORE
        # the prefix is applied. So we override the XML path but let the parent handle the rest.
        # However, MujocoXMLModel.__init__ applies prefix in its __init__, so we need to
        # inject sensors after super().__init__ and handle prefix ourselves.
        #
        # Strategy: We call super().__init__ with our own XML, then inject tactile sites/sensors
        # into the raw XML. Since add_prefix already ran, we store the un-prefixed names and
        # rely on the correct_naming mechanism for public access.
        #
        # Actually, looking at the code flow:
        #   MujocoXMLModel.__init__:
        #     1. sort_elements -> extracts _sensors list (un-prefixed names from XML)
        #     2. add_prefix -> modifies XML attributes with prefix
        #
        # So after super().__init__, _sensors contains the un-prefixed originals, and
        # the XML has been prefixed. To add new sensors correctly, we need to:
        #   1. Add un-prefixed site/sensor elements to the XML body/sensor elements
        #   2. Then manually apply prefix to the new elements
        #   3. Append the un-prefixed names to self._sensors
        #
        # Wait - actually the parent class (PandaGripperBase) hardcodes the XML path.
        # We need to override __init__ to use our own XML path. Let's use GripperModel directly.

        from robosuite.models.grippers.gripper_model import GripperModel
        GripperModel.__init__(self, xml_path_completion("grippers/panda_tactile_gripper.xml"), idn=idn)

        # Now inject tactile sensors into the loaded XML
        self._inject_tactile_sensors()

    def _inject_tactile_sensors(self):
        """
        Dynamically inject tactile sensor sites, touch sensor elements, and
        subdivision collision geoms into the already-loaded XML model.

        The original single pad collision geom is replaced with a grid of smaller
        collision geoms so that MuJoCo generates distributed contact points across
        the entire pad surface (not just at edges/corners).
        """
        # ---- Left fingerpad ----
        left_tip_body = find_elements(
            root=self.worldbody,
            tags="body",
            attribs={"name": f"{self.naming_prefix}finger_joint1_tip"},
            return_first=True,
        )
        assert left_tip_body is not None, "Could not find left fingertip body"

        # Remove original single pad collision geom; extract its contact properties
        left_pad_geom = find_elements(
            root=left_tip_body,
            tags="geom",
            attribs={"name": f"{self.naming_prefix}finger1_pad_collision"},
            return_first=True,
        )
        assert left_pad_geom is not None, "Could not find left pad collision geom"
        left_geom_attribs = dict(left_pad_geom.attrib)
        left_tip_body.remove(left_pad_geom)

        # Add grid of small collision geoms (distributes contact points)
        left_geom_names = subdivide_pad_collision(
            parent_body_element=left_tip_body,
            pad_center=[0.0, -0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="-y",
            prefix="left_pad",
            rows=self._tactile_rows,
            cols=self._tactile_cols,
            geom_attribs=left_geom_attribs,
        )

        # Add touch sensor sites and sensor elements (co-located with sub-geoms)
        left_names = generate_tactile_grid(
            parent_body_element=left_tip_body,
            sensor_element=self.sensor,
            pad_center=[0.0, -0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="-y",
            prefix="left_pad",
            rows=self._tactile_rows,
            cols=self._tactile_cols,
        )

        # ---- Right fingerpad ----
        right_tip_body = find_elements(
            root=self.worldbody,
            tags="body",
            attribs={"name": f"{self.naming_prefix}finger_joint2_tip"},
            return_first=True,
        )
        assert right_tip_body is not None, "Could not find right fingertip body"

        right_pad_geom = find_elements(
            root=right_tip_body,
            tags="geom",
            attribs={"name": f"{self.naming_prefix}finger2_pad_collision"},
            return_first=True,
        )
        assert right_pad_geom is not None, "Could not find right pad collision geom"
        right_geom_attribs = dict(right_pad_geom.attrib)
        right_tip_body.remove(right_pad_geom)

        right_geom_names = subdivide_pad_collision(
            parent_body_element=right_tip_body,
            pad_center=[0.0, 0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="+y",
            prefix="right_pad",
            rows=self._tactile_rows,
            cols=self._tactile_cols,
            geom_attribs=right_geom_attribs,
        )

        right_names = generate_tactile_grid(
            parent_body_element=right_tip_body,
            sensor_element=self.sensor,
            pad_center=[0.0, 0.005, -0.015],
            pad_halfsize=[0.008, 0.004, 0.008],
            normal_axis="+y",
            prefix="right_pad",
            rows=self._tactile_rows,
            cols=self._tactile_cols,
        )

        # ---- Apply naming prefix to ALL newly injected elements ----
        n_sensors = self._tactile_rows * self._tactile_cols

        # Prefix sub-geoms and sites in left tip body
        for elem in list(left_tip_body):
            if elem.tag in ("geom", "site"):
                name = elem.get("name", "")
                if name.startswith("left_pad_"):
                    elem.set("name", f"{self.naming_prefix}{name}")

        # Prefix sub-geoms and sites in right tip body
        for elem in list(right_tip_body):
            if elem.tag in ("geom", "site"):
                name = elem.get("name", "")
                if name.startswith("right_pad_"):
                    elem.set("name", f"{self.naming_prefix}{name}")

        # Prefix touch sensor elements (last 2*n_sensors in self.sensor)
        all_sensor_elems = list(self.sensor)
        for touch_elem in all_sensor_elems[-(2 * n_sensors):]:
            if touch_elem.tag == "touch":
                old_name = touch_elem.get("name")
                old_site = touch_elem.get("site")
                touch_elem.set("name", f"{self.naming_prefix}{old_name}")
                touch_elem.set("site", f"{self.naming_prefix}{old_site}")

        # Store un-prefixed names
        self._tactile_sensor_names["left"] = left_names
        self._tactile_sensor_names["right"] = right_names
        self._left_pad_geom_names = left_geom_names
        self._right_pad_geom_names = right_geom_names

        # Register in _sensors list so they appear in self.sensors property
        self._sensors.extend(left_names)
        self._sensors.extend(right_names)

        # Update _contact_geoms: remove deleted original pad geoms, add sub-geoms.
        # sort_elements() populated _contact_geoms before we removed the originals.
        self._contact_geoms = [
            g for g in self._contact_geoms
            if g not in ("finger1_pad_collision", "finger2_pad_collision")
        ]
        self._contact_geoms.extend(left_geom_names)
        self._contact_geoms.extend(right_geom_names)

    @property
    def tactile_grid_shape(self):
        """Shape of each tactile sensor grid: (rows, cols)."""
        return (self._tactile_rows, self._tactile_cols)

    @property
    def tactile_sensor_names(self):
        """
        Dictionary of un-prefixed tactile sensor names for each finger.

        Returns:
            dict: {"left": [list of str], "right": [list of str]}
                Each list has length rows*cols, in row-major order.
        """
        return self._tactile_sensor_names

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision"] + list(self._left_pad_geom_names),
            "right_finger": ["finger2_collision"] + list(self._right_pad_geom_names),
            "left_fingerpad": list(self._left_pad_geom_names),
            "right_fingerpad": list(self._right_pad_geom_names),
        }


class PandaTactileGripper(PandaTactileGripperBase):
    """
    Modifies PandaTactileGripperBase to only take one action (same as PandaGripper).
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
