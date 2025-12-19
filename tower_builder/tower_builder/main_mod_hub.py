#각도 안 들어간 ver

"""
🏗️ Smart Tower Builder - PRECISE STACKING
=========================================
[수정 완료]
1. 떨어뜨리기(Air Drop) 삭제 -> 바닥에 딱 맞춰서 안착
2. '짓눌림 방지' 완벽 해결:
   - 이전 블럭들의 높이를 누적 계산하여 정확한 Place Z 좌표 산출
   - 카메라 오차 보정 (36.1mm -> 30mm / 47.8mm -> 50mm 등으로 표준화)
3. 스레드, 좌표변환 등 기존 성공 로직 유지
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import threading
import time
import numpy as np

# 사용자 정의 모듈
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem

# ============================================================
# ⚙️ [설정]
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 150, 150

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            raise RuntimeError("Vision start failed")

        self.blocks = []
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None 
        self.is_working = False

        # [NEW] 쌓기 기록 (해체용)
        self.stack_history = []

        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1.0)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 연결 성공")
                self.gripper.move(0)
                time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()

    # ============================================================
    # [핵심] 좌표 변환 (성공했던 값 유지)
    # ============================================================
    def convert_camera_to_robot(self, cam_x_mm, cam_y_mm, cam_z_mm):
        final_x = 700 + cam_y_mm
        final_y = cam_x_mm + 5.0
        final_z = 820 - cam_z_mm
        if final_z <= 150.0:
            final_z = 150.0
        return final_x, final_y, final_z

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.is_working:
            print("⚠️ 로봇이 작업 중입니다! 잠시만 기다려주세요.")
            return

        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block and block not in self.selected_queue:
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                w = min(block.real_width_mm, block.real_height_mm)
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}] (크기: {w:.1f}mm)")
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥]을 클릭하세요.")

        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패. 다시 클릭해주세요.")
                return

            rs_x_mm = point_3d[0] * 1000
            rs_y_mm = point_3d[1] * 1000
            rs_z_mm = point_3d[2] * 1000

            final_x, final_y, final_z = self.convert_camera_to_robot(rs_x_mm, rs_y_mm, rs_z_mm)
            self.stack_base_coords = (final_x, final_y, final_z)

            print(f"📍 타워 위치(Base): X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
            worker = threading.Thread(target=self.execute_stacking_sequence, daemon=True)
            worker.start()

    def execute_stacking_sequence(self):
        from DSR_ROBOT2 import movel, movej, get_current_posx, wait
        from DR_common2 import posx, posj

        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작")

        stack_x, stack_y, stack_base_z = self.stack_base_coords
        current_stack_height = 0.0
        Rz_target = 90.0  # Pick / Place 공통 고정

        # [NEW] 이번 작업 기록 초기화
        self.stack_history = []

        try:
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

            for i, block in enumerate(self.selected_queue):
                measured_w = min(block.real_width_mm, block.real_height_mm)
                if measured_w >= 45.0:
                    real_block_height = 50.5
                    val_close = 580
                elif measured_w >= 30.0:
                    real_block_height = 40.7
                    val_close = 650
                else:
                    real_block_height = 30.5
                    val_close = 680

                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x, pick_y, pick_z = self.convert_camera_to_robot(cam_x, cam_y, cam_z)
                place_z = stack_base_z + current_stack_height + 1.0

                SAFE_Z = 350.0
                val_open = 0

                # [NEW] 해체를 위한 기록 (원래 자리 + 쌓은 자리)
                self.stack_history.append({
                    "pick_xyz": (pick_x, pick_y, pick_z),
                    "place_xyz": (stack_x, stack_y, place_z),
                    "val_close": val_close,
                    "Rz": Rz_target,
                })

                # ================= [PICK 동작] =================
                p_high = posx([pick_x, pick_y, SAFE_Z, 90, 180, Rz_target])
                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(3)

                self.gripper.move(val_open)
                wait(1)

                p_pick = posx([pick_x, pick_y, pick_z, 90, 180, Rz_target])
                movel(p_pick, vel=VELOCITY/2, acc=ACC/2)
                wait(2)

                self.gripper.move(val_close)
                wait(2)

                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(2)

                # ================= [PLACE 동작] =================
                p_place_high = posx([stack_x, stack_y, SAFE_Z, 90, 180, Rz_target])
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(2)

                p_place = posx([stack_x, stack_y, place_z, 90, 180, Rz_target])
                movel(p_place, vel=VELOCITY/2, acc=ACC/2)
                wait(2)

                self.gripper.move(val_open)
                wait(2)

                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(2)

                current_stack_height += real_block_height

            print("\n✨ 모든 작업 완료! 홈으로 이동.")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

        except Exception as e:
            self.get_logger().error(f"작업 중 오류 발생: {e}")
            try:
                print("🚨 오류 발생! 홈 위치로 복구 중...")
                home_pose = posj(0, 0, 90, 0, 90, 0)
                movej(home_pose, vel=VELOCITY, acc=ACC)
                wait(3)
                print("🏠 홈 위치 복구 완료")
            except Exception as recovery_error:
                self.get_logger().error(f"복구 중 오류 발생: {recovery_error}")

        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_working = False
            self.target_stack_count = 0
            print("🎉 완료! 다시 시작하려면 터미널을 확인하세요.")

    # ============================================================
    # [NEW] 역순 해체 시퀀스 (기록 기반, camera 안 씀)
    # ============================================================
    def execute_unstack_sequence(self):
        from DSR_ROBOT2 import movel, movej, wait
        from DR_common2 import posx, posj

        if not self.stack_history:
            print("⚠️ 해체할 기록이 없습니다. 먼저 탑을 쌓아주세요.")
            return

        self.is_working = True
        print("\n🧹 역순 해체 시퀀스 시작")

        SAFE_Z = 350.0
        val_open = 0

        try:
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

            # 역순으로 하나씩 해체
            for rec in reversed(self.stack_history):
                (pick_x, pick_y, pick_z) = rec["pick_xyz"]
                (place_x, place_y, place_z) = rec["place_xyz"]
                val_close = rec["val_close"]
                Rz_target = rec["Rz"]

                # ================= [UNSTACK PICK: 탑에서 집기] =================
                p_place_high = posx([place_x, place_y, SAFE_Z, 90, 180, Rz_target])
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(2)

                self.gripper.move(val_open)
                wait(1)

                # 살짝 위 여유 후 접근 (아래 블럭 간섭 방지)
                p_from_stack = posx([place_x, place_y, place_z + 1.0, 90, 180, Rz_target])
                movel(p_from_stack, vel=VELOCITY/2, acc=ACC/2)
                wait(2)

                self.gripper.move(val_close)
                wait(2)

                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(2)

                # ================= [UNSTACK PLACE: 원래 자리로 복귀] =================
                p_pick_high = posx([pick_x, pick_y, SAFE_Z, 90, 180, Rz_target])
                movel(p_pick_high, vel=VELOCITY, acc=ACC)
                wait(2)

                # 원래 pick_z에 “살포시” 내려놓기 (약간 여유)
                p_back = posx([pick_x, pick_y, pick_z + 1.0, 90, 180, Rz_target])
                movel(p_back, vel=VELOCITY/2, acc=ACC/2)
                wait(2)

                self.gripper.move(val_open)
                wait(2)

                movel(p_pick_high, vel=VELOCITY, acc=ACC)
                wait(2)

            print("\n✨ 해체 완료! 홈으로 이동.")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

            # 해체까지 완료했으면 기록 비우기
            self.stack_history = []

        except Exception as e:
            self.get_logger().error(f"해체 중 오류 발생: {e}")
            try:
                print("🚨 오류 발생! 홈 위치로 복구 중...")
                home_pose = posj(0, 0, 90, 0, 90, 0)
                movej(home_pose, vel=VELOCITY, acc=ACC)
                wait(3)
                print("🏠 홈 위치 복구 완료")
            except Exception as recovery_error:
                self.get_logger().error(f"복구 중 오류 발생: {recovery_error}")

        finally:
            self.is_working = False
            print("🎉 해체 시퀀스 종료")

    def process_and_render(self):
        if not self.vision.update():
            return
        display = self.vision.last_frame.copy()
        for block in self.vision.last_blocks:
            cv2.drawContours(display, [block.rotated_box], 0, (0,255,0), 2)
        cv2.imshow("Result", display)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    robot = RobotControllerNode()

    cv2.namedWindow("Result")
    cv2.setMouseCallback("Result", robot.mouse_callback)

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)

    threading.Thread(target=executor.spin, daemon=True).start()

    from DSR_ROBOT2 import movej, wait
    from DR_common2 import posj

    try:
        while rclpy.ok():
            # [NEW] 해체 트리거: idle 상태에서 'u' 입력하면 역순 해체
            if (not robot.is_working) and robot.stack_history:
                cmd = input("\n👉 (u) 해체 / (Enter) 계속 >> ").strip().lower()
                if cmd == "u":
                    worker = threading.Thread(target=robot.execute_unstack_sequence, daemon=True)
                    worker.start()

            if robot.target_stack_count == 0 and not robot.is_working:
                try:
                    home_pose = posj(0, 0, 90, 0, 90, 0)
                    movej(home_pose, vel=VELOCITY, acc=ACC)
                    wait(3)

                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 화면에서 선택하세요.")
                except ValueError:
                    pass

            robot.process_and_render()

    except KeyboardInterrupt:
        print("종료")

    finally:
        robot.terminate_gripper()
        robot.stop_camera()
        executor.shutdown()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()