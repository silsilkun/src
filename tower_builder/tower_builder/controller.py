"""
🤖 Robot Controller Module
===========================
Doosan 로봇 + 그리퍼 제어 모듈

BlockDetectionSystem과 함께 사용:
    from block_detection_system import BlockDetectionSystem, Block
    from robot_controller import RobotController
    
    with BlockDetectionSystem() as vision:
        robot = RobotController()
        robot.initialize()
        
        block = vision.get_closest_block()
        if block:
            robot.pick_block(block)
"""

import rclpy
from rclpy.node import Node
from typing import Optional, Tuple
from dataclasses import dataclass

# 로봇 설정
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"
DEFAULT_VELOCITY = 200
DEFAULT_ACC = 100


@dataclass
class RobotConfig:
    """로봇 설정"""
    robot_id: str = ROBOT_ID
    robot_model: str = ROBOT_MODEL
    velocity: int = DEFAULT_VELOCITY
    acceleration: int = DEFAULT_ACC
    
    # 카메라→로봇 좌표 변환 파라미터 (캘리브레이션 필요)
    offset_x: float = 685.0
    offset_y: float = 20.0
    offset_z: float = 810.0
    
    # 안전 높이
    min_z: float = 150.0
    approach_height: float = 50.0  # 접근 시 들어올리는 높이
    
    # 그리퍼 설정
    gripper_open: int = 100
    gripper_close: int = 550


class RobotController(Node):
    """
    Doosan 로봇 제어 클래스
    
    사용 예시:
        robot = RobotController()
        robot.initialize()
        
        # Block 객체로 직접 집기
        robot.pick_block(block)
        
        # 또는 3D 좌표로 이동
        robot.move_to(x_mm, y_mm, z_mm)
        
        robot.shutdown()
    """
    
    def __init__(self, config: Optional[RobotConfig] = None):
        super().__init__("robot_controller_node")
        
        self.config = config or RobotConfig()
        self.gripper = None
        self._is_initialized = False
        
        # DR_init은 main.py에서 설정됨 (여기서 중복 설정 불필요)
    
    # -------------------- 생명주기 --------------------
    
    def initialize(self, dsr_node=None) -> bool:
        """
        로봇 및 그리퍼 초기화
        
        Args:
            dsr_node: DR_init에 설정된 ROS2 노드 (main에서 전달)
        """
        try:
            from DSR_ROBOT2 import set_robot_mode, ROBOT_MODE_AUTONOMOUS, wait
            set_robot_mode(ROBOT_MODE_AUTONOMOUS)
            
            from tower_builder.gripper_drl_controller import GripperController
            # dsr_node가 있으면 그걸 사용, 없으면 self 사용
            node_to_use = dsr_node if dsr_node else self
            self.gripper = GripperController(node=node_to_use, namespace=self.config.robot_id)
            
            self.get_logger().info("Waiting for robot service...")
            wait(2)
            
            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed")
                return False
            
            # 그리퍼 열기
            self.gripper.move(self.config.gripper_open)
            
            self._is_initialized = True
            self.get_logger().info("✅ 로봇 초기화 완료")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 로봇 초기화 실패: {e}")
            return False
    
    def shutdown(self):
        """로봇 종료"""
        if self.gripper:
            self.gripper.terminate()
        self.get_logger().info("로봇 종료")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    # -------------------- 좌표 변환 --------------------
    
    def camera_to_robot(self, cam_x_mm: float, cam_y_mm: float, cam_z_mm: float
                       ) -> Tuple[float, float, float]:
        """
        카메라 좌표 → 로봇 좌표 변환
        
        Args:
            cam_x_mm, cam_y_mm, cam_z_mm: 카메라 기준 3D 좌표 (mm)
            
        Returns:
            (robot_x, robot_y, robot_z): 로봇 기준 좌표 (mm)
        """
        cfg = self.config
        
        # RealSense 좌표계 → 로봇 좌표계 변환
        # (이 부분은 실제 설치 상태에 따라 캘리브레이션 필요)
        robot_x = cfg.offset_x + cam_y_mm
        robot_y = cam_x_mm + cfg.offset_y
        robot_z = cfg.offset_z - cam_z_mm
        
        # 안전 높이 제한
        if robot_z < cfg.min_z:
            robot_z = cfg.min_z
        
        return (robot_x, robot_y, robot_z)
    
    def block_to_robot_coords(self, block) -> Optional[Tuple[float, float, float]]:
        """
        Block 객체에서 로봇 좌표 추출
        
        Args:
            block: Block 객체 (center_3d_mm 속성 필요)
            
        Returns:
            (robot_x, robot_y, robot_z) 또는 None
        """
        if not block.is_valid:
            return None
        
        cam_coords = block.center_3d_mm  # (x, y, z) in mm
        if cam_coords is None:
            return None
        
        return self.camera_to_robot(*cam_coords)
    
    # -------------------- 기본 동작 --------------------
    
    def move_to(self, x: float, y: float, z: float, 
                velocity: Optional[int] = None,
                acceleration: Optional[int] = None):
        """
        지정된 좌표로 이동 (현재 자세 유지)
        
        Args:
            x, y, z: 로봇 좌표 (mm)
        """
        from DSR_ROBOT2 import get_current_posx, movel, wait
        from DR_common2 import posx
        
        vel = velocity or self.config.velocity
        acc = acceleration or self.config.acceleration
        
        current_pos = get_current_posx()[0]
        _, _, _, rx, ry, rz = current_pos
        
        target = posx([x, y, z, rx, ry, rz])
        movel(target, vel=vel, acc=acc)
        wait(0.3)
    
    def move_home(self):
        """홈 자세로 이동"""
        from DSR_ROBOT2 import movej, wait
        from DR_common2 import posj
        
        home = posj(0, 0, 90, 0, 90, 0)
        movej(home, self.config.velocity, self.config.acceleration)
        wait(1.0)
        self.get_logger().info("🏠 홈 위치로 이동 완료")
    
    def gripper_open(self):
        """그리퍼 열기"""
        if self.gripper:
            self.gripper.move(self.config.gripper_open)
    
    def gripper_close(self):
        """그리퍼 닫기"""
        if self.gripper:
            self.gripper.move(self.config.gripper_close)
    
    # -------------------- 고수준 동작 --------------------
    
    def pick_block(self, block, return_home: bool = False) -> bool:
        """
        블록 집기
        
        Args:
            block: Block 객체
            return_home: 완료 후 홈으로 복귀 여부
            
        Returns:
            성공 여부
        """
        coords = self.block_to_robot_coords(block)
        if coords is None:
            self.get_logger().warn("블록 좌표를 가져올 수 없습니다")
            return False
        
        x, y, z = coords
        return self.pick_at(x, y, z, return_home)
    
    def pick_at(self, x: float, y: float, z: float, 
                return_home: bool = False) -> bool:
        """
        지정된 좌표에서 물체 집기
        
        Args:
            x, y, z: 로봇 좌표 (mm)
            return_home: 완료 후 홈으로 복귀 여부
        """
        from DSR_ROBOT2 import get_current_posx, movel, wait
        from DR_common2 import posx
        
        cfg = self.config
        vel, acc = cfg.velocity, cfg.acceleration
        
        try:
            current_pos = get_current_posx()[0]
            _, _, _, rx, ry, rz = current_pos
            
            z_up = z + cfg.approach_height
            
            # 1. 위에서 접근
            self.get_logger().info(f"📍 접근: ({x:.1f}, {y:.1f}, {z_up:.1f})")
            target_up = posx([x, y, z_up, rx, ry, rz])
            movel(target_up, vel=vel, acc=acc)
            wait(0.3)
            
            # 2. 하강
            self.get_logger().info(f"⬇️ 하강: ({x:.1f}, {y:.1f}, {z:.1f})")
            target_down = posx([x, y, z, rx, ry, rz])
            movel(target_down, vel=vel, acc=acc)
            wait(0.3)
            
            # 3. 그리퍼 닫기
            self.get_logger().info("✊ 집기")
            self.gripper_close()
            wait(1.5)
            
            # 4. 들어올리기
            movel(target_up, vel=vel, acc=acc)
            wait(0.3)
            
            if return_home:
                self.move_home()
            
            self.get_logger().info("✅ 집기 완료")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 집기 실패: {e}")
            return False
    
    def place_at(self, x: float, y: float, z: float,
                 return_home: bool = False) -> bool:
        """
        지정된 좌표에 물체 놓기
        
        Args:
            x, y, z: 로봇 좌표 (mm)
            return_home: 완료 후 홈으로 복귀 여부
        """
        from DSR_ROBOT2 import get_current_posx, movel, wait
        from DR_common2 import posx
        
        cfg = self.config
        vel, acc = cfg.velocity, cfg.acceleration
        
        try:
            current_pos = get_current_posx()[0]
            _, _, _, rx, ry, rz = current_pos
            
            z_up = z + cfg.approach_height
            
            # 1. 위에서 접근
            self.get_logger().info(f"📍 놓기 위치 접근: ({x:.1f}, {y:.1f}, {z_up:.1f})")
            target_up = posx([x, y, z_up, rx, ry, rz])
            movel(target_up, vel=vel, acc=acc)
            wait(0.3)
            
            # 2. 하강
            target_down = posx([x, y, z, rx, ry, rz])
            movel(target_down, vel=vel, acc=acc)
            wait(0.3)
            
            # 3. 그리퍼 열기
            self.get_logger().info("🖐️ 놓기")
            self.gripper_open()
            wait(1.0)
            
            # 4. 들어올리기
            movel(target_up, vel=vel, acc=acc)
            wait(0.3)
            
            if return_home:
                self.move_home()
            
            self.get_logger().info("✅ 놓기 완료")
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 놓기 실패: {e}")
            return False
    
    def pick_and_place(self, block, 
                       place_offset: Tuple[float, float, float] = (0, 100, 0),
                       return_home: bool = True) -> bool:
        """
        블록을 집어서 다른 위치에 놓기
        
        Args:
            block: Block 객체
            place_offset: 집은 위치 기준 놓을 위치 오프셋 (x, y, z) mm
            return_home: 완료 후 홈으로 복귀 여부
        """
        coords = self.block_to_robot_coords(block)
        if coords is None:
            self.get_logger().warn("블록 좌표를 가져올 수 없습니다")
            return False
        
        pick_x, pick_y, pick_z = coords
        place_x = pick_x + place_offset[0]
        place_y = pick_y + place_offset[1]
        place_z = pick_z + place_offset[2]
        
        # 집기
        if not self.pick_at(pick_x, pick_y, pick_z):
            return False
        
        # 놓기
        if not self.place_at(place_x, place_y, place_z, return_home):
            return False
        
        return True


# ============================================================
# ROS2 통합 (선택적)
# ============================================================

class RobotSystem:
    """
    ROS2 + 로봇 통합 관리 클래스
    
    rclpy 초기화/종료를 자동으로 처리
    
    사용 예시:
        with RobotSystem() as robot:
            robot.pick_at(300, 200, 150)
    """
    
    def __init__(self, config: Optional[RobotConfig] = None):
        self.config = config
        self.controller: Optional[RobotController] = None
        self._dsr_node = None
    
    def __enter__(self) -> RobotController:
        rclpy.init()
        
        # DSR 노드 생성
        cfg = self.config or RobotConfig()
        self._dsr_node = rclpy.create_node("dsr_node", namespace=cfg.robot_id)
        
        import DR_init
        DR_init.__dsr__node = self._dsr_node
        
        # 로봇 컨트롤러 생성 및 초기화
        self.controller = RobotController(self.config)
        if not self.controller.initialize():
            raise RuntimeError("로봇 초기화 실패")
        
        return self.controller
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.controller:
            self.controller.shutdown()
            self.controller.destroy_node()
        if self._dsr_node:
            self._dsr_node.destroy_node()
        rclpy.shutdown()
        return False


# ============================================================
# 테스트용 (독립 실행 시)
# ============================================================

if __name__ == "__main__":
    print("Robot Controller Module")
    print("이 모듈은 직접 실행하지 않고 import해서 사용하세요.")
    print()
    print("사용 예시:")
    print("  from robot_controller import RobotController, RobotSystem")
    print("  ")
    print("  with RobotSystem() as robot:")
    print("      robot.pick_at(300, 200, 150)")