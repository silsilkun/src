"""
🏗️ Camera Node - 블럭 감지 + UI + Publish (JSON 버전)
======================================================
[터미널 1에서 실행]
ros2 run tower_builder camera_node

기능:
1. RealSense로 블럭 감지
2. 마우스로 블럭 선택 + 타워 위치 선택
3. 선택 완료 시 /stack_command 토픽에 JSON publish
"""

import cv2
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from typing import List, Optional

from tower_builder.camera import BlockDetectionSystem, Block


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        
        # Publisher (std_msgs/String에 JSON 담아서 전송)
        self.stack_pub = self.create_publisher(String, "/stack_command", 10)
        
        # Vision System
        self.vision = BlockDetectionSystem()
        if not self.vision.start():
            self.get_logger().error("❌ 카메라 시작 실패!")
            raise RuntimeError("Vision start failed")
        
        # 상태 변수
        self.blocks: List[Block] = []
        self.target_stack_count = 0
        self.selected_queue: List[Block] = []
        self.stack_base_coords: Optional[tuple] = None
        self.is_command_sent = False
        
        # OpenCV 창 설정
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 640, 480)
        cv2.setMouseCallback("Camera", self.mouse_callback)
        
        # 타이머 (30fps 업데이트)
        self.timer = self.create_timer(1.0 / 30.0, self.update_loop)
        
        self.get_logger().info("✅ Camera Node 시작!")
        self.print_instructions()
    
    def print_instructions(self):
        """사용 안내 출력"""
        print("\n" + "=" * 50)
        print("🏗️ Smart Tower Builder - Camera Node")
        print("=" * 50)
        print("📌 사용법:")
        print("   1. 터미널에 탑 층수 입력")
        print("   2. 화면에서 블럭 클릭하여 선택")
        print("   3. 타워 놓을 바닥 위치 클릭")
        print("   4. 로봇 노드가 자동으로 작업 시작!")
        print("=" * 50)
        print("\n👉 몇 층 탑을 쌓으시겠습니까? (터미널에 숫자 입력)")
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 콜백"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.is_command_sent:
            print("⚠️ 이미 명령을 전송했습니다! 재시작하려면 'r'을 누르세요.")
            return
        
        if self.target_stack_count == 0:
            print("⚠️ 먼저 터미널에 탑 층수를 입력하세요!")
            return
        
        # 1단계: 블럭 선택
        if len(self.selected_queue) < self.target_stack_count:
            block = self.vision.find_block_at(x, y, update=False)
            if block:
                if block in self.selected_queue:
                    print("⚠️ 이미 선택한 블럭입니다.")
                    return
                
                self.selected_queue.append(block)
                block.selection_order = len(self.selected_queue)
                
                print(f"✅ 블럭 선택 [{len(self.selected_queue)}/{self.target_stack_count}]")
                print(f"   크기: {block.size_name} (측정: {block.min_side_mm:.1f}mm → 표준: {block.standardized_height}mm)")
                print(f"   그리퍼: {block.gripper_value}")
                
                if len(self.selected_queue) == self.target_stack_count:
                    print("\n🎯 블럭 선택 완료! 이제 [탑을 쌓을 바닥]을 클릭하세요.")
            else:
                print("⚠️ 블럭이 없는 위치입니다.")
        
        # 2단계: 타워 위치 선택 및 명령 전송
        elif self.stack_base_coords is None:
            point_3d = self.vision.get_3d_at(x, y)
            if point_3d is None or point_3d[2] == 0:
                print("⚠️ 바닥 인식 실패. 다시 클릭해주세요.")
                return
            
            self.stack_base_coords = (
                point_3d[0] * 1000,
                point_3d[1] * 1000,
                point_3d[2] * 1000
            )
            
            print(f"\n📍 타워 위치 선택됨!")
            print(f"   카메라 좌표: X={self.stack_base_coords[0]:.1f}, Y={self.stack_base_coords[1]:.1f}, Z={self.stack_base_coords[2]:.1f} mm")
            
            self.publish_stack_command()
    
    def publish_stack_command(self):
        """JSON으로 StackCommand publish"""
        # 블럭 정보를 딕셔너리 리스트로 변환
        blocks_data = []
        for block in self.selected_queue:
            block_dict = {
                "selection_order": block.selection_order,
                "center_x_mm": block.center_3d_mm[0] if block.center_3d_mm else 0,
                "center_y_mm": block.center_3d_mm[1] if block.center_3d_mm else 0,
                "center_z_mm": block.center_3d_mm[2] if block.center_3d_mm else 0,
                "real_width_mm": block.real_width_mm,
                "real_height_mm": block.real_height_mm,
                "standardized_height": block.standardized_height,
                "size_class": block.size_class,
                "gripper_value": block.gripper_value
            }
            blocks_data.append(block_dict)
        
        # 전체 명령 딕셔너리
        command_data = {
            "blocks": blocks_data,
            "target_x_mm": self.stack_base_coords[0],
            "target_y_mm": self.stack_base_coords[1],
            "target_z_mm": self.stack_base_coords[2]
        }
        
        # JSON 문자열로 변환하여 publish
        msg = String()
        msg.data = json.dumps(command_data)
        self.stack_pub.publish(msg)
        self.is_command_sent = True
        
        print("\n" + "=" * 50)
        print("📤 StackCommand 전송 완료!")
        print("=" * 50)
        print(f"   블럭 수: {len(blocks_data)}개")
        for bi in blocks_data:
            print(f"   #{bi['selection_order']}: 높이={bi['standardized_height']}mm, grip={bi['gripper_value']}")
        print(f"   타워 위치: ({command_data['target_x_mm']:.1f}, {command_data['target_y_mm']:.1f}, {command_data['target_z_mm']:.1f}) mm")
        print("=" * 50)
        print("\n🤖 로봇 노드에서 작업을 시작합니다...")
        print("   (재시작: 'r' 키)")
    
    def reset(self):
        """상태 초기화"""
        self.target_stack_count = 0
        self.selected_queue = []
        self.stack_base_coords = None
        self.is_command_sent = False
        print("\n🔄 초기화 완료! 층수를 다시 입력하세요.")
    
    def update_loop(self):
        """메인 업데이트 루프"""
        if not self.vision.update():
            return
        
        self.blocks = self.vision.last_blocks
        display = self.vision.last_frame.copy()
        
        for block in self.blocks:
            is_selected = (block in self.selected_queue)
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2
            
            cv2.drawContours(display, [block.rotated_box], 0, color, thickness)
            
            cx, cy = block.center_2d
            
            if is_selected and hasattr(block, 'selection_order') and block.selection_order > 0:
                cv2.putText(display, f"#{block.selection_order}", (cx - 10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if block.is_valid:
                txt = f"{block.standardized_height:.0f}mm"
                cv2.putText(display, txt, (cx - 20, cy + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 상태 메시지
        if self.target_stack_count == 0:
            status = "Input tower height in terminal"
        elif self.is_command_sent:
            status = "Command sent! Press 'r' to reset"
        elif len(self.selected_queue) < self.target_stack_count:
            status = f"Select blocks ({len(self.selected_queue)}/{self.target_stack_count})"
        elif self.stack_base_coords is None:
            status = "Click target floor position"
        else:
            status = "Ready"
        
        cv2.putText(display, status, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display, f"Blocks: {len(self.blocks)}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Camera", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self.get_logger().info("종료 요청...")
            rclpy.shutdown()
        elif key == ord('r'):
            self.reset()


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraNode()
    
    import threading
    
    def input_thread():
        while rclpy.ok():
            if node.target_stack_count == 0 and not node.is_command_sent:
                try:
                    val = input("\n👉 몇 층 탑? >> ")
                    cnt = int(val)
                    if cnt > 0:
                        node.target_stack_count = cnt
                        print(f"✅ {cnt}개 블럭을 화면에서 선택하세요.")
                except ValueError:
                    print("⚠️ 숫자를 입력하세요.")
                except EOFError:
                    break
    
    input_th = threading.Thread(target=input_thread, daemon=True)
    input_th.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.vision.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()