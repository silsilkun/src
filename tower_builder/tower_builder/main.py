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
VELOCITY, ACC = 60, 60  # 속도 살짝 낮춰서 정확도 UP

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

        # 그리퍼 초기화
        self.gripper = None
        try:
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            time.sleep(1.0)
            if self.gripper.initialize():
                self.get_logger().info("✅ 그리퍼 연결 성공")
                # 초기화: 0 = Open (지나님 피셜)
                self.gripper.move(0) 
                time.sleep(1.0)
        except Exception as e:
            self.get_logger().error(f"그리퍼 오류: {e}")

    def stop_camera(self):
        self.vision.stop()

    def terminate_gripper(self):
        if self.gripper: self.gripper.terminate()

    # ============================================================
    # [핵심] 좌표 변환 (성공했던 값 유지)
    # ============================================================
    def convert_camera_to_robot(self, cam_x_mm, cam_y_mm, cam_z_mm):
        # 1. X축: 635 + Y - 20 (강사님 공식)
        final_x = 690 + cam_y_mm #- 15
        
        # 2. Y축: X 그대로 (강사님 공식)
        final_y = cam_x_mm
        
        # 3. Z축: 825 - Z (지나님 수정 값! 너무 깊게 안 박히게)
        final_z = 823 - cam_z_mm

        # 안전 바닥 높이 제한
        if final_z <= 150.0: 
            final_z = 150.0
            
        return final_x, final_y, final_z

    # ============================================================
    # 마우스 콜백
    # ============================================================
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_working:
            print("⚠️ 로봇이 작업 중입니다! 잠시만 기다려주세요.")
            return

        # 1. 블럭 선택
        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block:
                if block in self.selected_queue: return
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                
                w = min(block.real_width_mm, block.real_height_mm)
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}] (크기: {w:.1f}mm)")
                
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! [탑을 쌓을 바닥]을 클릭하세요.")

        # 2. 위치 지정 및 실행
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패. 다시 클릭해주세요.")
                return

            rs_x_mm = point_3d[0] * 1000
            rs_y_mm = point_3d[1] * 1000
            rs_z_mm = point_3d[2] * 1000

            # 좌표 변환 적용
            final_x, final_y, final_z = self.convert_camera_to_robot(rs_x_mm, rs_y_mm, rs_z_mm)
            
            self.stack_base_coords = (final_x, final_y, final_z)
            print(f"📍 타워 위치(Base): X={final_x:.1f}, Y={final_y:.1f}, Z={final_z:.1f}")
            
            worker = threading.Thread(target=self.execute_stacking_sequence)
            worker.daemon = True
            worker.start()

    # ============================================================
    # 실행 시퀀스 (높이 누적 계산 적용)
    # ============================================================
    def execute_stacking_sequence(self):
        self.is_working = True
        print("\n🚀 로봇 작업 시퀀스 시작")
        
        stack_x, stack_y, stack_base_z = self.stack_base_coords
        
        # [NEW] 현재까지 쌓인 높이를 저장하는 변수 (0부터 시작)
        current_stack_height = 0.0

        from DSR_ROBOT2 import movel, movej, get_current_posx, wait
        from DR_common2 import posx, posj

        try:
            # 1. 홈 정렬
            print("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

            # 2. 적재 루프
            for i, block in enumerate(self.selected_queue):
                # --------------------------------------------------------
                # [높이 계산 로직]
                # 카메라 측정값(w)을 실제 규격(30, 40, 50)으로 변환
                # (오차 때문에 47.8mm 이렇게 나오면 50mm로 인식하게 함)
                # --------------------------------------------------------
                measured_w = min(block.real_width_mm, block.real_height_mm)
                
                if measured_w >= 45.0:
                    real_block_height = 50.0 # 대형
                    val_close = 580
                    size_name = "대형(5cm)"
                elif measured_w >= 35.0:
                    real_block_height = 40.0 # 중형
                    val_close = 650
                    size_name = "중형(4cm)"
                else:
                    real_block_height = 30.0 # 소형
                    val_close = 680
                    size_name = "소형(3cm)"

                print(f"\n🏗️ [{i+1}층 작업 시작] 블럭: {size_name} (실측: {measured_w:.1f}mm) =======")

                # --- 좌표 계산 ---
                cam_x, cam_y, cam_z = block.center_3d_mm
                pick_x, pick_y, pick_z = self.convert_camera_to_robot(cam_x, cam_y, cam_z)
                
                # [중요] 놓을 높이 = 바닥 높이 + 지금까지 쌓인 높이
                # (이렇게 해야 2층, 3층이 정확히 그 위에 안착함)
                # +1.0mm는 아주 미세한 안전 여유 (종이 한 장 두께) -> 짓누름 방지용
                place_z = stack_base_z + current_stack_height + 1.0

                # # --- 접근 방향 및 Rz 계산 ---
                # cur_pos = get_current_posx()[0]
                # cur_x, cur_y = cur_pos[0], cur_pos[1]
                # cur_rx, cur_ry = cur_pos[3], cur_pos[4]

                # dx, dy = pick_x - cur_x, pick_y - cur_y
                # if abs(dx) > abs(dy):
                #     approach_axis = "x+" if dx > 0 else "x-"
                # else:
                #     approach_axis = "y+" if dy > 0 else "y-"

                # # Rz 설정
                # if approach_axis == "x+":   Rz_target = 180.0
                # elif approach_axis == "x-": Rz_target = 0.0
                # elif approach_axis == "y+": Rz_target = -90.0
                # elif approach_axis == "y-": Rz_target = 90.0
                
                # print(f"   🧭 접근 방향: {approach_axis} -> Rz: {Rz_target}")

                SAFE_Z = 350.0 
                val_open = 0 

                # ================= [PICK 동작] =================
                print("   🚀 [1] 상공 이동 (XY축 정렬)")
                p_high = posx([pick_x, pick_y, SAFE_Z, cur_rx, cur_ry, Rz_target])
                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(3.5)

                if self.gripper: self.gripper.move(val_open)
                wait(2)

                print("   🔻 [2] 목표 상공 진입 (z+50)")
                p_ready = posx([pick_x, pick_y, pick_z + 50, cur_rx, cur_ry, Rz_target])
                movel(p_ready, vel=VELOCITY, acc=ACC)
                wait(2)

                print(f"   🔻 [3] Pick 하강 (Z={pick_z:.1f})")
                p_pick = posx([pick_x, pick_y, pick_z, cur_rx, cur_ry, Rz_target])
                movel(p_pick, vel=VELOCITY/2, acc=ACC/2)
                wait(3)

                print(f"   ✊ [4] 그립 ({val_close})")
                if self.gripper: self.gripper.move(val_close)
                wait(3)

                print("   🔼 [5] 상승")
                movel(p_high, vel=VELOCITY, acc=ACC)
                wait(3)

                # ================= [PLACE 동작] =================
                print(f"   🚀 [6] Place 상공 이동 (목표 높이: {place_z:.1f})")
                p_place_high = posx([stack_x, stack_y, SAFE_Z, cur_rx, cur_ry, Rz_target])
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(3)

                print(f"   🔻 [7] Place 하강")
                # 정확히 계산된 place_z로 이동 (툭 떨어뜨리기 아님, 살포시 안착)
                p_place = posx([stack_x, stack_y, place_z, cur_rx, cur_ry, Rz_target])
                movel(p_place, vel=VELOCITY/2, acc=ACC/2)
                wait(3)

                print("   🖐 [8] 놓기")
                if self.gripper: self.gripper.move(val_open)
                wait(4)

                print("   🔼 [9] 복귀")
                movel(p_place_high, vel=VELOCITY, acc=ACC)
                wait(5)
                
                # [중요] 블럭 하나 쌓았으니, 다음 블럭을 위해 높이 누적
                current_stack_height += real_block_height
                print(f"   📈 현재 탑 높이: {current_stack_height:.1f}mm (다음 블럭은 이 위에 쌓음)")

            print("\n✨ 모든 작업 완료! 홈으로 이동.")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)

        except Exception as e:
            self.get_logger().error(f"작업 중 오류 발생: {e}")
        finally:
            self.selected_queue = []
            self.stack_base_coords = None
            self.is_working = False
            self.target_stack_count = 0 
            print("🎉 완료! 다시 시작하려면 터미널을 확인하세요.")

    def process_and_render(self):
        if not self.vision.update(): return

        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        
        for block in self.blocks:
            is_sel = (block in self.selected_queue)
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.drawContours(display, [block.rotated_box], 0, col, 2)
            
            if hasattr(block, 'selection_order'):
                cx, cy = block.center_2d
                cv2.putText(display, f"#{block.selection_order}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if self.target_stack_count == 0: msg = "Input Number in Terminal"
        elif self.is_working: msg = "Robot Working..."
        elif len(self.selected_queue) < self.target_stack_count: msg = f"Select Blocks ({len(self.selected_queue)}/{self.target_stack_count})"
        elif self.stack_base_coords is None: msg = "Click Target Floor"
        else: msg = "Ready"
        
        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.imshow("Result", display)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    except: pass

    robot = RobotControllerNode()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.setMouseCallback("Result", robot.mouse_callback)

    print("\n" + "="*40)
    print("🏗️ Smart Tower Builder (PRECISE STACKING)")
    print("="*40)

    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(dsr_node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if robot.target_stack_count == 0 and not robot.is_working:
                try:
                    val = input("\n👉 몇 층 탑을 쌓으시겠습니까? (숫자) >> ")
                    cnt = int(val)
                    if cnt > 0:
                        robot.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 화면에서 선택하세요.")
                except ValueError: pass
            
            robot.process_and_render()
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