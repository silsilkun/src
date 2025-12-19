import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

import DR_init
from tower_builder.gripper_drl_controller import GripperController

from tower_builder.camera_try import BlockDetectionSystem 

ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 100, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    
    def __init__(self):
        super().__init__("robot_controller_node")

        # ---- 그리퍼 초기화 (레드팬더님 코드 유지) ----
        self.gripper = None
        try:
            from DSR_ROBOT2 import wait
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            self.get_logger().info("Waiting for service /dsr01/drl/drl_start...")
            wait(2)
            if not self.gripper.initialize():
                raise Exception("Gripper initialization failed")
            self.get_logger().info("그리퍼를 활성화합니다...")
            self.gripper.move(0)
        except Exception as e:
            self.get_logger().error(f"Gripper setup error: {e}")
        
        # --- 상태/저장 변수 ---
        self.stage = "PICK_BLOCK"   # "PICK_BLOCK" -> "SET_TOWER"
        self.pick_robot_xyz = None  # (x,y,z)
        self.Rz_turned = None       # 블럭 각도+100
        self.tower_robot_xyz = None # (tower_x, tower_y, tower_z)
        self.grip = None            # 선택된 블럭에 따른 그리퍼 닫힘 값
        self.total_block_count = 0
        self.current_block_index = 0

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    #camera_try에서 넘어온 블럭 데이터로 로봇 실행
    def on_block_clicked(self, block):
        # 1) 블럭 중심 카메라 좌표
        X, Y, Z = block.center_cam_mm

        # 2) 카메라->로봇 변환(레드팬더님 기존식 유지)
        x = 685 + Y
        y = X + 20
        z = 811 - Z
        if z <= 150:
            z = 150

        # 3) Rz_turned = (블럭 각도) + 100
        # (camera_try.py에서 angle_deg는 "세로 기준 0도"로 이미 변환된 값)
        self.Rz_turned = float(block.angle_deg) + 100.0
        self.pick_robot_xyz = (x, y, z)

        # 블럭 크기
        grip_map = {
            "SMALL_3cm": 600,
            "MEDIUM_4cm": 550,
            "LARGE_5cm": 500,
        }
        self.grip = grip_map[block.label]


        self.get_logger().info(
            f"[PICK] label={block.label} side={block.side_mm:.1f}mm "
            f"angle={block.angle_deg:.1f}deg -> Rz_turned={self.Rz_turned:.1f}deg | "
            f"robot_xyz=({x:.1f},{y:.1f},{z:.1f})"
        )

        # 다음 클릭은 "놓을 위치"로 받도록 상태 변경
        self.stage = "SET_TOWER"
        print("\n[STEP] 이제 '놓을 위치(빈 곳)'를 클릭하세요.\n")

        self.current_block_index += 1
        print(f"[PROGRESS] {self.current_block_index} / {self.total_block_count}")

        if self.current_block_index >= self.total_block_count:
            print("\n[FINISH] 모든 블럭 선택 완료. 탑 쌓기를 시작할 수 있습니다.\n")


    def on_point_clicked(self, cam_xyz_mm):
        """
        camera_try.py에서 '빈 곳 클릭'에도 호출될 콜백
        cam_xyz_mm: (X,Y,Z) in mm (카메라 좌표)
        """
        if self.stage != "SET_TOWER":
            return

        X, Y, Z = cam_xyz_mm

        # 카메라->로봇 변환(동일 식)
        tower_x = 685 + Y
        tower_y = X + 20
        tower_z = 811 - Z
        if tower_z <= 150:
            tower_z = 150

        self.tower_robot_xyz = (tower_x, tower_y, tower_z)

        print(
            f"[TOWER] cam(X,Y,Z)=({X:.1f},{Y:.1f},{Z:.1f})mm -> "
            f"robot(tower_x,y,z)=({tower_x:.1f},{tower_y:.1f},{tower_z:.1f})"
        )

        px, py, pz = self.pick_robot_xyz
        self.move_robot_and_control_gripper(px, py, pz)

        print("\n[OK] 픽업/타워 좌표가 모두 준비되었습니다.\n")
        self.stage = "PICK_BLOCK"

    def move_robot_and_control_gripper(self, x, y, z):
        from DSR_ROBOT2 import get_current_posx, movel, wait, movej
        from DR_common2 import posx, posj

        try:
            safe_zone = 200
            Rz_turned = self.Rz_turned
            tx, ty, tz = self.tower_robot_xyz


            # 홈 위치로 이동
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej((home_pose), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 그리퍼 열기
            self.gripper.move(0)
            wait(1.0)

            # 블럭 상단으로 이동
            block_high = posx [x, y, z + safe_zone, 100, 180, Rz_turned]
            movel((block_high), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 블럭 위치 대기
            block_at = posx [x, y, z, 90, 180, Rz_turned]
            movel((block_at), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 블럭 집기
            self.gripper.move(int(self.grip))
            wait(1.5)

            # 블럭 들기
            block_high = posx [x, y, z + safe_zone, 100, 180, Rz_turned]
            movel((block_high), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 지정위치 상단으로 이동
            block_tower = posx [tx, ty, tz + safe_zone, 100, 180, 100]
            movel((block_tower), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 탑 쌓을 위치로 이동
            block_tower = posx [tx, ty, tz, 100, 180, 100]
            movel((block_tower), vel=VELOCITY, acc=ACC)
            wait(0.3)

            # 블럭 놓기
            self.gripper.move(0)
            wait(1.0)

            # 홈 위치로 이동
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej((home_pose), vel=VELOCITY, acc=ACC)
            wait(0.3)

        except Exception as e:
            self.get_logger().error(f"로봇 이동 및 그리퍼 제어 중 오류 발생: {e}")

def main(args=None):
    rclpy.init(args=args)

    try:
        block_count = int(input("쌓을 블럭 갯수를 입력하세요: "))
    except ValueError:
        print("숫자만 입력하세요.")
        rclpy.shutdown()
        return

    print(f"[INFO] 총 {block_count}개의 블럭을 쌓습니다.\n")

    # dsr_node 세팅(레드팬더님 코드 유지)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    robot = RobotControllerNode()
    robot.total_block_count = block_count
    cam = BlockDetectionSystem()  # camera_try 노드 생성
    cam.on_block_clicked = robot.on_block_clicked
    cam.on_point_clicked = robot.on_point_clicked

    executor = MultiThreadedExecutor()
    executor.add_node(dsr_node)
    executor.add_node(robot)
    executor.add_node(cam)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        robot.terminate_gripper()
        cam.destroy_node()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()