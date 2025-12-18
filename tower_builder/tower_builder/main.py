"""
🏗️ Smart Tower Builder - NUCLEAR OPTION
=======================================
[최후의 수단 적용]
1. 그리퍼 전원: os.system()으로 터미널 명령어 직접 주입
2. 동작 스킵 방지: movel 대신 movej 사용
3. 디버깅 로그: 좌표 출력
"""

import os
import cv2
import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# 사용자 정의 모듈
import DR_init
from tower_builder.gripper_drl_controller import GripperController
from tower_builder.camera import BlockDetectionSystem

# ============================================================
# ⚙️ 설정
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 100, 50

TRANSFORM_OFFSET_X = 685.0
TRANSFORM_OFFSET_Y = 20.0
CAMERA_Z_HEIGHT = 810.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")
        
        # 비전 시스템 초기화
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            raise RuntimeError("Vision start failed")

        self.blocks = []
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None
        self.is_working = False

        # 그리퍼 초기화
        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 객체 생성됨")
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    # ---------------------------
    # 카메라/그리퍼 종료
    # ---------------------------
    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: 
            self.gripper.terminate()

    # ---------------------------
    # [필살기 1] 그리퍼 전원 강제 공급
    # ---------------------------
    def force_gripper_power_on(self):
        print("\n⚡ [강제 명령] 그리퍼 전원 24V 공급 시도 중...")
        cmd = f"ros2 service call /{ROBOT_ID}/tool/set_tool_voltage dsr_msgs/srv/SetToolVoltage \"{{voltage: 24}}\""
        result = os.system(cmd)
        print("✅ 전원 공급 명령 전송 완료!" if result == 0 else "⚠️ 전원 공급 명령 실패")
        time.sleep(1)

    # ---------------------------
    # 마우스 클릭 이벤트
    # ---------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.is_working:
            if self.is_working: print("⚠️ 로봇이 움직이고 있습니다!")
            return

        # 1. 블럭 선택
        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block and block not in self.selected_queue:
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                w = min(block.real_width_mm, block.real_height_mm)
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}] (크기: {w:.1f}mm)")

                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥] 클릭")

        # 2. 타워 위치 지정
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if not point_3d or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패.")
                return

            cam_x_mm = point_3d[0] * 1000
            cam_y_mm = point_3d[1] * 1000
            self.stack_base_coords = (TRANSFORM_OFFSET_X + cam_y_mm, cam_x_mm + TRANSFORM_OFFSET_Y)
            print(f"📍 타워 위치: ({self.stack_base_coords[0]:.1f}, {self.stack_base_coords[1]:.1f})")

            threading.Thread(target=self.execute_stacking_sequence, daemon=True).start()

    # ---------------------------
    # 메인 스택 시퀀스
    # ---------------------------
    def execute_stacking_sequence(self):
        from DSR_ROBOT2 import movej, movel, wait
        from DR_common2 import posj, posx

        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작")

        BASE_Z, BLOCK_H = 152.0, 40.0
        stack_x, stack_y = self.stack_base_coords

        try:
            # 1. 전원 강제 공급
            self.force_gripper_power_on()

            # 2. 그리퍼 워밍업
            if self.gripper:
                print("✊ 그리퍼 테스트 동작...")
                for p in [0, 800, 0]:
                    self.gripper.move(p)
                    wait(0.5)

            # 3. 홈 위치 이동
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(1.0)

            # 4. 블럭 적재
            for i, block in enumerate(self.selected_queue):
                print(f"\n🏗️ [{i+1}층 작업 시작] ---------------------")
                self.pick_and_place_block(block, stack_x, stack_y, BASE_Z, BLOCK_H, i)

            print("\n✨ 작업 완료! 홈 복귀.")
            movej(home_pose, vel=VELOCITY, acc=ACC)

        except Exception as e:
            self.get_logger().error(f"실행 중 오류: {e}")
        finally:
            self.selected_queue.clear()
            self.stack_base_coords = None
            self.is_working = False

    # ---------------------------
    # 개별 블럭 픽앤플레이스
    # ---------------------------
    def pick_and_place_block(self, block, stack_x, stack_y, base_z, block_h, layer):
        from DSR_ROBOT2 import get_current_posx, movel, wait
        from DR_common2 import posx

        # 좌표 계산
        cam_x, cam_y, cam_z = block.center_3d_mm
        px = TRANSFORM_OFFSET_X + cam_y
        py = cam_x + TRANSFORM_OFFSET_Y
        pz = max(CAMERA_Z_HEIGHT - cam_z, base_z)
        sx, sy, sz = stack_x, stack_y, base_z + layer * block_h

        width = min(block.real_width_mm, block.real_height_mm)
        target_open, target_close = (0, 500)
        if width <= 35: target_open, target_close = 300, 850
        elif width <= 45: target_open, target_close = 200, 600
        else: target_open, target_close = 0, 350

        cur_x = get_current_posx()[0]
        rx, ry, rz = cur_x[3], cur_x[4], cur_x[5]
        safe_z = 350.0

        print(f"   👉 PICK 좌표: {px:.1f}, {py:.1f}, {pz:.1f}")
        print(f"   👉 PLACE 좌표: {sx:.1f}, {sy:.1f}, {sz:.1f}")

        # Pick
        movel(posx([px, py, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        wait(0.2)
        if self.gripper: self.gripper.move(target_open)
        movel(posx([px, py, pz, rx, ry, rz]), vel=VELOCITY/2, acc=ACC/2)
        wait(0.5)
        if self.gripper: self.gripper.move(target_close)
        print("   ✊ [3] 그립!")
        wait(1.5)
        movel(posx([px, py, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)

        # Place
        movel(posx([sx, sy, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)
        movel(posx([sx, sy, sz + 15, rx, ry, rz]), vel=VELOCITY/2, acc=ACC/2)
        wait(0.5)
        if self.gripper: self.gripper.move(0)
        print("   🖐 [6] 놓기 완료")
        wait(0.8)
        movel(posx([sx, sy, safe_z, rx, ry, rz]), vel=VELOCITY, acc=ACC)

    # ---------------------------
    # 비전 프레임 처리 및 렌더링
    # ---------------------------
    def process_and_render(self):
        cfg = self.vision.config
        cfg.threshold = cv2.getTrackbarPos("Threshold", "Control")
        cfg.min_area = cv2.getTrackbarPos("Min Area", "Control")
        cfg.max_area = cv2.getTrackbarPos("Max Area", "Control")

        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        cv2.rectangle(display, (cfg.roi_x, cfg.roi_y), (cfg.roi_x+cfg.roi_w, cfg.roi_y+cfg.roi_h), (0,0,255), 2)

        for block in self.blocks:
            col = (0, 255, 255) if block in self.selected_queue else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            cx, cy = block.center_2d
            w_mm = min(block.real_width_mm, block.real_height_mm)
            cv2.putText(display, f"{w_mm:.0f}mm", (cx-20, cy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if hasattr(block, 'selection_order'):
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        msg = "Input number in terminal"
        if self.target_stack_count > 0:
            if len(self.selected_queue) < self.target_stack_count: msg = "Select Blocks..."
            elif self.stack_base_coords is None: msg = ">> Click Target Floor <<"
            else: msg = "Auto Stacking..."
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)

        if self.vision.last_depth is not None:
            depth_view = cv2.applyColorMap(cv2.convertScaleAbs(self.vision.last_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_view)


# ============================================================
# 메인 루프
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)
    threading.Thread(target=executor.spin, daemon=True).start()

    # OpenCV 윈도우 및 트랙바
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)
    
    cv2.namedWindow("Control")
    cfg = robot.vision.config
    cv2.createTrackbar("Threshold", "Control", cfg.threshold, 255, lambda x: None)
    cv2.createTrackbar("Min Area", "Control", cfg.min_area, 5000, lambda x: None)
    cv2.createTrackbar("Max Area", "Control", cfg.max_area, 30000, lambda x: None)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (NUCLEAR OPTION)")
    print("="*40)

    try:
        while rclpy.ok():
            if robot.target_stack_count == 0:
                try:
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 선택하세요.")
                except ValueError: pass
                continue

            robot.process_and_render()

            if not robot.is_working and robot.target_stack_count > 0 and robot.stack_base_coords is not None:
                if len(robot.selected_queue) == 0:
                    robot.target_stack_count = 0
                    robot.stack_base_coords = None
                    print("\n🎉 완료! 다시 시작합니다.")

            if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        print("종료")
    finally:
        robot.terminate_gripper()
        robot.stop_camera()
        cv2.destroyAllWindows()
        executor.shutdown()
        robot.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
