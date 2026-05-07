from ._base_task import Base_Task
from .utils import *


class lift_empty_cup(Base_Task):
    """Grasp empty cup and lift; success by cup height (no place phase)."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def get_info(self):
        return {"{A}": "021_cup/base0"}

    def load_actors(self):
        tag = np.random.randint(0, 2)
        cup_xlim = [[0.15, 0.3], [-0.3, -0.15]]
        self.cup = rand_create_actor(
            self,
            xlim=cup_xlim[tag],
            ylim=[-0.2, 0.05],
            modelname="021_cup",
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
            convex=True,
            model_id=0,
        )
        self.add_prohibit_area(self.cup, padding=0.05)
        self.delay(2)
        self.init_cup_z = float(self.cup.get_functional_point(0, "pose").p[2])

    def play_once(self):
        cup_pose = self.cup.get_pose().p
        arm_tag = ArmTag("right" if cup_pose[0] > 0 else "left")

        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag,
                pre_grasp_dis=0.1,
                contact_point_id=[0, 2][int(arm_tag == "left")],
            ))
        self.move(self.move_by_displacement(arm_tag, z=0.10, move_axis="arm"))
        self.delay(8)

        self.info["info"] = {"{A}": "021_cup/base0"}
        return self.info

    def check_success(self):
        cup_z = float(self.cup.get_functional_point(0, "pose").p[2])
        init_z = float(getattr(self, "init_cup_z", cup_z))
        return cup_z - init_z > 0.06
