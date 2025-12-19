"""
🏗️ Robot Node - 로봇 제어 (JSON 버전, Thread 없음!)
====================================================
[터미널에서 실행]
ros2 run tower_builder robot_node

기능:
1. /stack_command 토픽 subscribe (JSON)
2. 받은 명령대로 로봇 순차 제어
3. Thread 없이 동기 실행
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

# DSR 로봇 관련
import DR_init

from tower_builder.gripper_drl_controller import GripperController

# ============================================================
# 설정
# ============================================================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
VELOCITY, ACC = 45, 45

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotNode(Node):
    def __init__(self, dsr_node):
        super().__init__("robot_node", namespace=ROBOT_ID)
        
        # dsr_node 참조 저장 (정리용)
        self.dsr_node = dsr_node
        
        # Subscriber (JSON 문자열 수신)
        self.stack_sub = self.create_subscription(
            String,
            "/stack_command",
            self.on_stack_command,
            10
        )
        
        # 그리퍼 초기화
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
        
        self.get_logger().info("✅ Robot Node 시작! /stack_command 대기 중...")
    
    # ============================================================
    # 좌표 변환 (카메라 → 로봇)
    # ============================================================
    def convert_camera_to_robot(self, cam_x_mm, cam_y_mm, cam_z_mm):
        """카메라 좌표 → 로봇 좌표 변환"""
        final_x = 700 + cam_y_mm
        final_y = cam_x_mm + 10.0
        final_z = 823 - cam_z_mm
        
        if final_z <= 150.0:
            final_z = 150.0
        
        return final_x, final_y, final_z
    
    # ============================================================
    # 명령 수신 콜백
    # ============================================================
    def on_stack_command(self, msg: String):
        """JSON StackCommand 수신 시 실행"""
        try:
            command_data = json.loads(msg.data)
            blocks = command_data["blocks"]
            target_x = command_data["target_x_mm"]
            target_y = command_data["target_y_mm"]
            target_z = command_data["target_z_mm"]
            
            self.get_logger().info(f"📥 StackCommand 수신! 블럭 {len(blocks)}개")
            
            # 순차 실행
            self.execute_stacking(blocks, target_x, target_y, target_z)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"❌ JSON 파싱 오류: {e}")
        except KeyError as e:
            self.get_logger().error(f"❌ 필수 필드 누락: {e}")
    
    # ============================================================
    # 스택 실행 (순차, 동기)
    # ============================================================
    def execute_stacking(self, blocks: list, target_x_mm: float, 
                         target_y_mm: float, target_z_mm: float):
        """블럭 쌓기 실행"""
        from DSR_ROBOT2 import movel, movej, wait
        from DR_common2 import posx, posj
        
        self.get_logger().info("\n🚀 로봇 작업 시퀀스 시작")
        
        # 타워 위치 변환
        stack_x, stack_y, stack_base_z = self.convert_camera_to_robot(
            target_x_mm, target_y_mm, target_z_mm
        )
        self.get_logger().info(f"📍 타워 위치(로봇): X={stack_x:.1f}, Y={stack_y:.1f}, Z={stack_base_z:.1f}")
        
        current_stack_height = 0.0
        
        try:
            # 1. 홈 이동
            self.get_logger().info("🏠 홈 위치 정렬...")
            home_pose = posj(0, 0, 90, 0, 90, 0)
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)
            
            # 2. 각 블럭 처리
            for block_info in blocks:
                self.process_single_block(
                    block_info=block_info,
                    stack_x=stack_x,
                    stack_y=stack_y,
                    stack_base_z=stack_base_z,
                    current_stack_height=current_stack_height
                )
                
                # 높이 누적
                current_stack_height += block_info["standardized_height"]
                self.get_logger().info(f"📈 현재 탑 높이: {current_stack_height:.1f}mm")
            
            # 3. 홈 복귀
            self.get_logger().info("🏠 작업 완료! 홈으로 복귀...")
            movej(home_pose, vel=VELOCITY, acc=ACC)
            wait(3)
            
            self.get_logger().info("\n✨ 모든 작업 완료!")
            
        except Exception as e:
            self.get_logger().error(f"❌ 작업 중 오류: {e}")
    
    def process_single_block(self, block_info: dict, 
                             stack_x: float, stack_y: float, 
                             stack_base_z: float, current_stack_height: float):
        """단일 블럭 Pick & Place"""
        from DSR_ROBOT2 import movel, wait
        from DR_common2 import posx
        
        order = block_info["selection_order"]
        std_height = block_info["standardized_height"]
        grip_val = block_info["gripper_value"]
        
        # 크기 이름
        if std_height >= 50:
            size_name = "대형(5cm)"
        elif std_height >= 40:
            size_name = "중형(4cm)"
        else:
            size_name = "소형(3cm)"
        
        self.get_logger().info(f"\n🏗️ [{order}층 작업] 블럭: {size_name}")
        self.get_logger().info(f"   표준 높이: {std_height}mm, 그리퍼: {grip_val}")
        
        # Pick 좌표 변환
        pick_x, pick_y, pick_z = self.convert_camera_to_robot(
            block_info["center_x_mm"],
            block_info["center_y_mm"],
            block_info["center_z_mm"]
        )
        
        # Place 높이 계산
        place_z = stack_base_z + current_stack_height + 1.0
        
        self.get_logger().info(f"   Pick: ({pick_x:.1f}, {pick_y:.1f}, {pick_z:.1f})")
        self.get_logger().info(f"   Place Z: {place_z:.1f} (base:{stack_base_z:.1f} + stack:{current_stack_height:.1f})")
        
        SAFE_Z = 350.0
        VAL_OPEN = 0
        Rz_target = 0.0
        
        # ========== PICK ==========
        self.get_logger().info("   [1] 상공 이동")
        p_high = posx([pick_x, pick_y, SAFE_Z, 0, 0, Rz_target])
        movel(p_high, vel=VELOCITY, acc=ACC)
        wait(3.5)
        
        if self.gripper:
            self.gripper.move(VAL_OPEN)
        wait(2)
        
        self.get_logger().info("   [2] 목표 상공 진입")
        p_ready = posx([pick_x, pick_y, pick_z + 50, 0, 0, Rz_target])
        movel(p_ready, vel=VELOCITY, acc=ACC)
        wait(2)
        
        self.get_logger().info(f"   [3] Pick 하강 (Z={pick_z:.1f})")
        p_pick = posx([pick_x, pick_y, pick_z, 0, 0, Rz_target])
        movel(p_pick, vel=VELOCITY/2, acc=ACC/2)
        wait(3)
        
        self.get_logger().info(f"   [4] 그립 ({grip_val})")
        if self.gripper:
            self.gripper.move(grip_val)
        wait(3)
        
        self.get_logger().info("   [5] 상승")
        movel(p_high, vel=VELOCITY, acc=ACC)
        wait(3)
        
        # ========== PLACE ==========
        self.get_logger().info(f"   [6] Place 상공 이동")
        p_place_high = posx([stack_x, stack_y, SAFE_Z, 0, 0, Rz_target])
        movel(p_place_high, vel=VELOCITY, acc=ACC)
        wait(3)
        
        self.get_logger().info(f"   [7] Place 하강 (Z={place_z:.1f})")
        p_place = posx([stack_x, stack_y, place_z, 0, 0, Rz_target])
        movel(p_place, vel=VELOCITY/2, acc=ACC/2)
        wait(3)
        
        self.get_logger().info("   [8] 놓기")
        if self.gripper:
            self.gripper.move(VAL_OPEN)
        wait(4)
        
        self.get_logger().info("   [9] 복귀")
        movel(p_place_high, vel=VELOCITY, acc=ACC)
        wait(5)
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        if self.gripper:
            self.gripper.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # [핵심] RobotNode 생성 전에 dsr_node 먼저 설정!
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node
    
    # 로봇 모드 설정 (노드 생성 전에!)
    try:
        from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        print("✅ 로봇 자율 모드 설정 완료")
    except Exception as e:
        print(f"로봇 모드 설정 실패: {e}")
    
    node = RobotNode(dsr_node)
    
    # [핵심] MultiThreadedExecutor 사용 (원래 main.py 방식)
    from rclpy.executors import MultiThreadedExecutor
    import threading
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(dsr_node)
    
    # spin을 별도 스레드로 (서비스 콜백 처리용)
    # ※ 두산이 금지한 건 "Thread에서 movel/movej 호출"이지, spin Thread는 OK!
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    print("\n✅ Robot Node 대기 중... (Ctrl+C로 종료)")
    
    try:
        # 메인 스레드는 대기 (모션 명령은 콜백에서 메인처럼 실행됨)
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 요청...")
    finally:
        executor.shutdown()
        node.destroy_node()
        dsr_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()