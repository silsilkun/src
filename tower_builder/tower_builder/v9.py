import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Literal
from contextlib import contextmanager
import json
from datetime import datetime

# ===== Global System Handle =====
_system = None

# ===== 1. 깊이 샘플링 설정 =====

@dataclass
class DepthSamplingConfig:
    """깊이 유효성 및 샘플링 설정"""
    
    # 유효한 깊이 범위 (미터)
    min_valid_depth: float = 0.05   # 5cm - 카메라 최소 감지 거리
    max_valid_depth: float = 3.0    # 3m - 카메라 최대 감지 거리
    
    # 샘플링할 때의 범위
    sampling_min_depth: float = 0.05
    sampling_max_depth: float = 3.0
    
    # 샘플링 활성화
    enable_sampling: bool = True
    
    # 샘플링 거리 (픽셀)
    sampling_distances: List[int] = field(default_factory=lambda: [5, 10, 15])
    
    # 샘플링 방향 (4-way 또는 8-way)
    sampling_directions: Literal[4, 8] = 4  # 4-way: 상하좌우, 8-way: 대각선 포함
    
    # 샘플링 방법 (중앙값 또는 평균)
    sampling_method: Literal['median', 'mean'] = 'median'
    
    def get_sampling_offsets(self) -> List[Tuple[int, int]]:
        """설정에 따라 샘플링 오프셋 생성"""
        offsets = []
        for dist in self.sampling_distances:
            # 상하좌우
            offsets.extend([
                (-dist, 0), (dist, 0),
                (0, -dist), (0, dist),
            ])
            # 대각선 (8-way일 때만)
            if self.sampling_directions == 8:
                offsets.extend([
                    (-dist, -dist), (dist, -dist),
                    (-dist, dist), (dist, dist),
                ])
        return offsets


# ===== 2. 이미지 처리 설정 =====

@dataclass
class ImageProcessingConfig:
    """이미지 처리 파라미터"""
    
    # Gaussian Blur 커널 크기 (홀수만 가능)
    blur_kernel_size: int = 5  # 5x5 커널
    
    # 모폴로지 연산 커널 크기 (홀수만 가능)
    morph_kernel_size: int = 3  # 3x3 커널
    
    # Contour 근사 (곡선을 얼마나 단순화할지, 0-1)
    contour_approx_epsilon: float = 0.04  # 호의 길이의 4%
    
    # 꼭지점 개수 범위
    min_vertices: int = 3   # 삼각형 이상
    max_vertices: int = 8   # 8각형 이하


# ===== 3. 카메라 초기화 설정 =====

@dataclass
class CameraWarmupConfig:
    """카메라 초기화 설정"""
    warmup_frames: int = 30  # 30프레임 @ 30FPS = 1초


# ===== 4. 컨투어 필터 설정 =====

@dataclass
class ContourFilterConfig:
    """컨투어 필터링 기준"""
    
    # 면적 필터 (픽셀²)
    min_area: int = 90      # 너무 작은 블록 제외
    max_area: int = 4000    # 너무 큰 객체 제외
    
    # 종횡비 필터
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 3.0
    
    # Solidity 필터
    min_solidity: float = 0.7


# ===== 5. 깊이 필터 설정 =====

@dataclass
class DepthFilterConfig:
    """3D 깊이 기반 필터"""
    
    min_depth: float = 0.1   # 10cm 이상
    max_depth: float = 2.0   # 2m 이하


# ===== 6. ROI 설정 =====

@dataclass
class ROIConfig:
    """관심 영역(Region of Interest) 설정"""
    
    x: int = 190      # 좌상단 X 좌표
    y: int = 140      # 좌상단 Y 좌표
    width: int = 230  # 가로
    height: int = 180 # 세로


# ===== 7. 캐시 설정 =====

@dataclass
class CacheConfig:
    """메모리 캐시 관리 설정"""
    
    max_cached_frames: int = 1      # 최신 1개만 유지
    max_clicked_blocks: int = 0     # 0 = 무제한
    max_clicked_floor_points: int = 0


# ===== 8. 통합 Detector 설정 =====

@dataclass
class DetectorConfig:
    """블록 감지기 전체 설정"""
    
    # 이진화 임계값 (0-255)
    threshold: int = 200
    
    # 각 서브 설정
    roi: ROIConfig = field(default_factory=ROIConfig)
    contour_filter: ContourFilterConfig = field(default_factory=ContourFilterConfig)
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    depth_sampling: DepthSamplingConfig = field(default_factory=DepthSamplingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    camera_warmup: CameraWarmupConfig = field(default_factory=CameraWarmupConfig)


# 데이터 클래스

@dataclass
class Block:
    """감지된 블록 정보를 담는 데이터 클래스"""
    
    # 기본 정보
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center_2d: Tuple[int, int]       # 픽셀 좌표 (x, y)
    contour: np.ndarray = field(compare=False, repr=False)
    rotated_box: np.ndarray = field(compare=False, repr=False)
    
    # 형태 정보
    area: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    angle: float = 0.0  # 회전 각도 (도)
    
    # 3D 정보
    center_3d: Optional[Tuple[float, float, float]] = None  # (X, Y, Z) 미터
    depth: float = 0.0  # 미터
    
    # 실제 크기 (mm)
    real_width_mm: float = 0.0
    real_height_mm: float = 0.0
    
    # 메타 정보
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    click_order: int = 0  # 클릭 순서
    
    # -------------------- 편의 속성 --------------------
    
    @property
    def side_length_mm(self) -> float:
        """한 변의 평균 길이 (mm)"""
        return (self.real_width_mm + self.real_height_mm) / 2
    
    @property
    def depth_cm(self) -> float:
        """깊이 (cm)"""
        return self.depth * 100
    
    @property
    def depth_mm(self) -> float:
        """깊이 (mm)"""
        return self.depth * 1000
    
    @property
    def center_3d_mm(self) -> Optional[Tuple[float, float, float]]:
        """3D 좌표 (mm 단위)"""
        if self.center_3d is None:
            return None
        x, y, z = self.center_3d
        return (x * 1000, y * 1000, z * 1000)
    
    @property
    def is_valid(self) -> bool:
        """유효한 깊이 정보가 있는지"""
        return self.depth > 0
        
    def copy_with_click_order(self, order: int) -> "Block":
        """클릭 순서를 포함한 안전한 Block 복사"""
        data = asdict(self)
        data["contour"] = self.contour
        data["rotated_box"] = self.rotated_box
        data["click_order"] = order
        return Block(**data)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (저장용)"""
        return {
            'click_order': self.click_order,
            'timestamp': self.timestamp,
            'center_2d_x': self.center_2d[0],
            'center_2d_y': self.center_2d[1],
            'bbox_x': self.bbox[0],
            'bbox_y': self.bbox[1],
            'bbox_w': self.bbox[2],
            'bbox_h': self.bbox[3],
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'solidity': self.solidity,
            'angle': self.angle,
            'depth_m': self.depth,
            'depth_cm': self.depth_cm,
            'depth_mm': self.depth_mm,
            'center_3d_x_mm': self.center_3d_mm[0] if self.center_3d_mm else None,
            'center_3d_y_mm': self.center_3d_mm[1] if self.center_3d_mm else None,
            'center_3d_z_mm': self.center_3d_mm[2] if self.center_3d_mm else None,
            'real_width_mm': self.real_width_mm,
            'real_height_mm': self.real_height_mm,
            'side_length_mm': self.side_length_mm,
        }
    
    def __str__(self) -> str:
        if self.is_valid:
            return (f"Block(order={self.click_order}, center={self.center_2d}, "
                    f"depth={self.depth_cm:.1f}cm, "
                    f"size={self.side_length_mm:.1f}mm)")
        return f"Block(order={self.click_order}, center={self.center_2d}, no depth)"

# 바닥 클릭 데이터 클래스

@dataclass
class FloorPoint:
    """바닥(빈 공간) 클릭 정보를 저장"""

    pixel: Tuple[int, int]
    depth: float
    point_3d: Optional[Tuple[float, float, float]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def point_3d_mm(self):
        if self.point_3d is None:
            return None
        x, y, z = self.point_3d
        return (x * 1000, y * 1000, z * 1000)


# 카메라 클래스

class RealSenseCamera:
    """RealSense 카메라 제어 클래스"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._intrinsics: Optional[rs.intrinsics] = None
        self._depth_scale: float = 0.001
        self._is_running: bool = False
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def intrinsics(self) -> Optional[rs.intrinsics]:
        return self._intrinsics
    
    @property
    def depth_scale(self) -> float:
        return self._depth_scale
    
    def start(self, warmup_config: Optional[CameraWarmupConfig] = None) -> bool:
        """카메라 시작"""
        if self._is_running:
            return True
        
        if warmup_config is None:
            warmup_config = CameraWarmupConfig()
            
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, 
                               rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, 
                               rs.format.z16, self.fps)
            
            profile = self._pipeline.start(config)
            self._align = rs.align(rs.stream.color)
            
            # Depth scale 가져오기
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            
            # Intrinsics 가져오기
            depth_stream = profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # 워밍업
            print("📷 카메라 초기화 중...")
            for _ in range(warmup_config.warmup_frames):
                self._pipeline.wait_for_frames()
            
            self._is_running = True
            print(f"✅ RealSense 시작! ({self.width}x{self.height})")
            print(f"   Depth Scale: {self._depth_scale:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False
    
    def stop(self):
        """카메라 정지"""
        if self._pipeline and self._is_running:
            self._pipeline.stop()
            self._is_running = False
            print("📷 카메라 정지")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """정렬된 컬러/뎁스 프레임 반환"""
        if not self._is_running:
            return None, None
            
        try:
            frames = self._pipeline.wait_for_frames()
            aligned = self._align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception:
            return None, None
    
    def get_depth_at(self, x: int, y: int, depth_image: np.ndarray, 
                     config: DepthSamplingConfig) -> float:
        """특정 픽셀의 깊이값 반환 (미터)"""
        x, y = int(x), int(y)
        
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        # 중심점 값
        raw_depth = depth_image[y, x]
        depth_m = raw_depth * self._depth_scale
        
        if depth_m > config.min_valid_depth:
            return depth_m
        
        if not config.enable_sampling:
            return 0.0
        
        # 주변 샘플링
        offsets = config.get_sampling_offsets()
        
        valid_depths = []
        for dx, dy in offsets:
            sx, sy = x + dx, y + dy
            if 0 <= sx < self.width and 0 <= sy < self.height:
                raw = depth_image[sy, sx]
                d = raw * self._depth_scale
                if config.sampling_min_depth < d < config.sampling_max_depth:
                    valid_depths.append(d)
        
        if valid_depths:
            valid_depths.sort()
            if config.sampling_method == 'median':
                return valid_depths[len(valid_depths) // 2]
            else:  # mean
                return sum(valid_depths) / len(valid_depths)
        
        return 0.0
    
    def pixel_to_3d(self, x: int, y: int, depth_image: np.ndarray,
                   config: DepthSamplingConfig) -> Optional[Tuple[float, float, float]]:
        """픽셀 좌표를 3D 좌표로 변환 (미터)"""
        depth = self.get_depth_at(x, y, depth_image, config)
        
        if depth <= 0 or self._intrinsics is None:
            return None
        
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], depth)
        return (point[0], point[1], depth)
    
    def calc_real_size(self, width_px: float, height_px: float, 
                       depth: float) -> Tuple[float, float]:
        """픽셀 크기를 실제 크기(mm)로 변환"""
        if depth <= 0 or self._intrinsics is None:
            return (0.0, 0.0)
        
        real_w = (width_px * depth * 1000) / self._intrinsics.fx
        real_h = (height_px * depth * 1000) / self._intrinsics.fy
        return (real_w, real_h)


# 감지기 클래스

class BlockDetector:
    """블록 감지기"""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._binary_view: Optional[np.ndarray] = None
    
    @property
    def binary_view(self) -> Optional[np.ndarray]:
        """최근 이진화 이미지 (디버깅용)"""
        return self._binary_view
    
    @property
    def roi(self) -> Tuple[int, int, int, int]:
        """현재 ROI (x, y, w, h)"""
        c = self.config.roi
        return (c.x, c.y, c.width, c.height)
    
    def detect(self, frame: np.ndarray, depth_image: np.ndarray,
               camera: RealSenseCamera) -> List[Block]:
        """프레임에서 블록 감지"""
        cfg = self.config
        roi_cfg = cfg.roi
        img_cfg = cfg.image_processing
        blocks = []
        
        # ROI 추출
        roi = frame[roi_cfg.y:roi_cfg.y+roi_cfg.height, 
                   roi_cfg.x:roi_cfg.x+roi_cfg.width]
        
        # 전처리
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, 
                                   (img_cfg.blur_kernel_size, 
                                    img_cfg.blur_kernel_size), 0)
        _, binary = cv2.threshold(blurred, cfg.threshold, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (img_cfg.morph_kernel_size, 
                                           img_cfg.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self._binary_view = binary
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            block = self._process_contour(cnt, depth_image, camera)
            if block is not None:
                blocks.append(block)
        
        return blocks
    
    def _process_contour(self, cnt: np.ndarray, depth_image: np.ndarray,
                         camera: RealSenseCamera) -> Optional[Block]:
        """단일 컨투어 처리"""
        cfg = self.config
        cf = cfg.contour_filter
        df = cfg.depth_filter
        img_cfg = cfg.image_processing
        roi_cfg = cfg.roi
        
        # 면적 필터
        area = cv2.contourArea(cnt)
        if not (cf.min_area < area < cf.max_area):
            return None
        
        # 회전 사각형
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (_, _), (w, h), angle = rect
        
        if h == 0 or w == 0:
            return None
        
        # 종횡비 필터
        aspect = max(w, h) / min(w, h)
        if not (cf.min_aspect_ratio <= aspect <= cf.max_aspect_ratio):
            return None
        
        # Solidity 필터
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = area / hull_area
        if solidity < cf.min_solidity:
            return None
        
        # 꼭지점 수 필터
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, img_cfg.contour_approx_epsilon * peri, True)
        if not (img_cfg.min_vertices <= len(approx) <= img_cfg.max_vertices):
            return None
        
        # 전역 좌표로 변환
        box_global = box.copy()
        box_global[:, 0] += roi_cfg.x
        box_global[:, 1] += roi_cfg.y
        
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += roi_cfg.x
        cnt_global[:, :, 1] += roi_cfg.y
        
        # 중심점 계산
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + roi_cfg.x
        cy = int(M["m01"] / M["m00"]) + roi_cfg.y
        
        # 바운딩 박스
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Block 생성
        block = Block(
            bbox=(x + roi_cfg.x, y + roi_cfg.y, bw, bh),
            center_2d=(cx, cy),
            contour=cnt_global,
            rotated_box=box_global,
            area=area,
            aspect_ratio=aspect,
            solidity=solidity,
            angle=angle
        )
        
        # 3D 정보 추가
        point_3d = camera.pixel_to_3d(cx, cy, depth_image, cfg.depth_sampling)
        
        if point_3d:
            block.center_3d = point_3d
            block.depth = point_3d[2]
            
            if df.min_depth < block.depth < df.max_depth:
                real_w, real_h = camera.calc_real_size(w, h, block.depth)
                block.real_width_mm = real_w
                block.real_height_mm = real_h
        
        return block


# 통합 시스템 클래스

class BlockDetectionSystem:
    
    def __init__(self, 
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 config: Optional[DetectorConfig] = None):

        self._camera = RealSenseCamera(camera_width, camera_height, camera_fps)
        self._detector = BlockDetector(config)
        
        # 캐시
        self._last_frame: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_blocks: List[Block] = []
        
        # 클릭한 블록 저장 리스트
        self._clicked_blocks: List[Block] = []
        self._clicked_floor_points: List[FloorPoint] = []
    
    def __enter__(self) -> "BlockDetectionSystem":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def start(self) -> bool:
        """시스템 시작"""
        return self._camera.start(self._detector.config.camera_warmup)
    
    def stop(self):
        """시스템 정지"""
        self._camera.stop()
        cv2.destroyAllWindows()
    
    @property
    def is_running(self) -> bool:
        return self._camera.is_running
    
    @property
    def config(self) -> DetectorConfig:
        """감지기 설정"""
        return self._detector.config
    
    @config.setter
    def config(self, value: DetectorConfig):
        self._detector.config = value
    
    @property
    def camera(self) -> RealSenseCamera:
        """카메라 인스턴스 (고급 사용)"""
        return self._camera
    
    @property
    def detector(self) -> BlockDetector:
        """감지기 인스턴스 (고급 사용)"""
        return self._detector
    
    def update(self) -> bool:
        """새 프레임을 가져와서 블록 감지 수행"""
        color, depth = self._camera.get_frames()
        if color is None:
            return False
        
        self._last_frame = color
        self._last_depth = depth
        self._last_blocks = self._detector.detect(color, depth, self._camera)
        return True
    
    def get_blocks(self, update: bool = True) -> List[Block]:
        """감지된 블록 리스트 반환"""
        if update:
            self.update()
        return self._last_blocks.copy()
    
    def get_valid_blocks(self, update: bool = True) -> List[Block]:
        """유효한 깊이 정보가 있는 블록만 반환"""
        blocks = self.get_blocks(update)
        return [b for b in blocks if b.is_valid]
    
    def _is_already_clicked(self, block: Block) -> bool:
        """이미 클릭된 블록인지 확인"""
        for b in self._clicked_blocks:
            if b.bbox == block.bbox:
                return True
        return False
    
    def get_clicked_blocks(self) -> List[Block]:
        """클릭한 블록 리스트 반환"""
        return self._clicked_blocks.copy()
    
    def get_clicked_floor_points(self) -> List[FloorPoint]:
        """클릭한 바닥 포인트 리스트 반환"""
        return self._clicked_floor_points.copy()
    
    def clear_clicked_blocks(self):
        """클릭한 블록 리스트 초기화"""
        self._clicked_blocks.clear()
        self._clicked_floor_points.clear()
        print("🗑️  클릭 블록 리스트 초기화됨")
    
    def print_clicked_blocks_summary(self):
        """클릭한 블록들의 요약 정보 출력"""
        if not self._clicked_blocks:
            print("⚠️  클릭한 블록이 없습니다")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 클릭한 블록 요약 (총 {len(self._clicked_blocks)}개)")
        print("=" * 60)
        
        for i, block in enumerate(self._clicked_blocks, 1):
            print(f"\n[{i}] {block}")
            if block.is_valid and block.center_3d_mm:
                x, y, z = block.center_3d_mm
                print(f"    위치: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                print(f"    크기: {block.side_length_mm:.1f} mm")
                print(f"    각도: {block.angle:.1f} °")
        
        print("=" * 60 + "\n")

    def print_clicked_floor_points_summary(self):
        """클릭한 바닥 포인트 요약 출력"""
        if not self._clicked_floor_points:
            print("⚠️  클릭한 바닥 포인트가 없습니다")
            return

        print("\n" + "=" * 60)
        print(f"🟦 클릭한 바닥 포인트 요약 (총 {len(self._clicked_floor_points)}개)")
        print("=" * 60)

        for i, fp in enumerate(self._clicked_floor_points, 1):
            print(f"\n[{i}] 픽셀 좌표: {fp.pixel}")

            if fp.point_3d_mm:
                x, y, z = fp.point_3d_mm
                print(f"    3D 좌표: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
                print(f"    깊이: {fp.depth * 100:.1f} cm")
            else:
                print("    ⚠️ 깊이 정보 없음")

        print("=" * 60)
    
    def get_closest_block(self, update: bool = True) -> Optional[Block]:
        """가장 가까운 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.depth)
    
    def get_farthest_block(self, update: bool = True) -> Optional[Block]:
        """가장 먼 블록 반환"""
        blocks = self.get_valid_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.depth)
    
    def get_largest_block(self, update: bool = True) -> Optional[Block]:
        """가장 큰 블록 반환"""
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.area)
    
    def get_smallest_block(self, update: bool = True) -> Optional[Block]:
        """가장 작은 블록 반환"""
        blocks = self.get_blocks(update)
        if not blocks:
            return None
        return min(blocks, key=lambda b: b.area)
    
    def get_block_count(self, update: bool = True) -> int:
        """감지된 블록 수"""
        return len(self.get_blocks(update))
    
    def find_blocks_in_depth_range(self,
                                   min_depth: float = 0,
                                   max_depth: float = float('inf'),
                                   update: bool = True) -> List[Block]:
        """특정 깊이 범위의 블록들 반환"""
        blocks = self.get_valid_blocks(update)
        return [b for b in blocks if min_depth <= b.depth <= max_depth]
    
    def find_block_at(self, x: int, y: int, 
                      update: bool = False) -> Optional[Block]:
        """특정 픽셀 위치의 블록 반환"""
        blocks = self.get_blocks(update)
        for block in blocks:
            bx, by, bw, bh = block.bbox
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return block
        return None
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """마지막 컬러 프레임"""
        return self._last_frame
    
    @property
    def last_depth(self) -> Optional[np.ndarray]:
        """마지막 깊이 프레임"""
        return self._last_depth
    
    @property
    def last_blocks(self) -> List[Block]:
        """마지막 감지 결과"""
        return self._last_blocks.copy()
    
    def get_depth_at(self, x: int, y: int) -> float:
        """특정 픽셀의 깊이값"""
        if self._last_depth is None:
            return 0.0
        return self._camera.get_depth_at(x, y, self._last_depth, 
                                         self._detector.config.depth_sampling)
    
    def get_3d_at(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """특정 픽셀의 3D 좌표"""
        if self._last_depth is None:
            return None
        return self._camera.pixel_to_3d(x, y, self._last_depth,
                                        self._detector.config.depth_sampling)
    
    def run_debug(self):
        """디버그 GUI 실행"""
        print("\n" + "=" * 50)
        print("🏗️ Block Detection - Debug Mode")
        print("=" * 50)
        print("📌 조작:")
        print("   - 블록 클릭: 상세 정보 + 리스트 저장")
        print("   - 빈 공간 클릭: 깊이 확인")
        print("   - 'p' 키: 저장된 블록 요약")
        print("   - 'c' 키: 저장 리스트 초기화")
        print("   - ESC: 종료")
        print("=" * 50 + "\n")
        
        selected_idx = -1
        
        def on_mouse(event, x, y, flags, param):
            nonlocal selected_idx

            if event != cv2.EVENT_LBUTTONDOWN:
                return

            for i, block in enumerate(self._last_blocks):
                bx, by, bw, bh = block.bbox
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_idx = i

                    if self._is_already_clicked(block):
                        print("⚠️ 이미 클릭된 블록입니다")
                        return

                    block_copy = block.copy_with_click_order(
                        len(self._clicked_blocks) + 1
                    )

                    self._clicked_blocks.append(block_copy)
                    self._print_block_info(block_copy)

                    print(f"💾 블록 저장 완료 (총 {len(self._clicked_blocks)}개)")
                    return

            selected_idx = -1

            depth = self.get_depth_at(x, y)
            point_3d = self.get_3d_at(x, y)

            floor_point = FloorPoint(
                pixel=(x, y),
                depth=depth,
                point_3d=point_3d
            )

            self._clicked_floor_points.append(floor_point)

            print("\n🟦 바닥 클릭 저장")
            print(f"  픽셀: ({x}, {y})")

            if point_3d:
                X, Y, Z = floor_point.point_3d_mm
                print(f"  3D 좌표: X={X:.1f}mm Y={Y:.1f}mm Z={Z:.1f}mm")
            else:
                print("  깊이 없음")

            print(f"  총 바닥 클릭 수: {len(self._clicked_floor_points)}")
        
        cv2.namedWindow("Result")
        cv2.setMouseCallback("Result", on_mouse)
        
        cv2.namedWindow("Control")
        cv2.createTrackbar("Threshold", "Control", 
                          self.config.threshold, 255, lambda x: None)
        cv2.createTrackbar("Min Area", "Control", 
                          self.config.contour_filter.min_area, 5000, lambda x: None)
        cv2.createTrackbar("Max Area", "Control", 
                          self.config.contour_filter.max_area, 30000, lambda x: None)
        
        try:
            while True:
                # 트랙바 값 적용
                self.config.threshold = cv2.getTrackbarPos("Threshold", "Control")
                self.config.contour_filter.min_area = cv2.getTrackbarPos("Min Area", "Control")
                self.config.contour_filter.max_area = cv2.getTrackbarPos("Max Area", "Control")
                
                if not self.update():
                    continue
                
                display = self._draw_result(selected_idx)
                
                cv2.putText(display, f"Clicked: {len(self._clicked_blocks)}", 
                           (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.imshow("Result", display)
                
                if self._detector.binary_view is not None:
                    cv2.imshow("Binary (ROI)", self._detector.binary_view)
                
                depth_display = self._draw_depth()
                cv2.imshow("Depth", depth_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    break
                elif key == ord('p'):
                    self.print_clicked_blocks_summary()
                    self.print_clicked_floor_points_summary()
                elif key == ord('c'):
                    self.clear_clicked_blocks()
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self._clicked_blocks:
                print(f"\n💾 {len(self._clicked_blocks)}개 블록이 저장되어 있습니다")
            
            cv2.destroyAllWindows()
            print("👋 디버그 모드 종료")
    
    def _draw_result(self, selected_idx: int = -1) -> np.ndarray:
        """결과 이미지 그리기"""
        display = self._last_frame.copy()
        roi_cfg = self.config.roi
        
        cv2.rectangle(display, 
                     (roi_cfg.x, roi_cfg.y),
                     (roi_cfg.x + roi_cfg.width, roi_cfg.y + roi_cfg.height),
                     (0, 0, 255), 2)
        cv2.putText(display, "ROI (WHITE)", (roi_cfg.x, roi_cfg.y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for i, block in enumerate(self._last_blocks):
            is_selected = (i == selected_idx)
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2
            
            cv2.drawContours(display, [block.rotated_box], 0, color, thickness)
            cx, cy = block.center_2d
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            
            if block.is_valid:
                txt_depth = f"{block.depth_cm:.0f}cm"
                cv2.putText(display, txt_depth, (cx - 15, cy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                txt_size = f"{block.side_length_mm:.0f}mm"
                cv2.putText(display, txt_size, (cx - 20, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            else:
                cv2.putText(display, "no depth", (cx - 25, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            cv2.putText(display, f"({cx},{cy})", (cx - 25, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(display, f"Blocks: {len(self._last_blocks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display
    
    def _draw_depth(self) -> np.ndarray:
        """깊이 이미지 시각화"""
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(self._last_depth, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        valid_mask = self._last_depth > 0
        if np.any(valid_mask):
            min_d = np.min(self._last_depth[valid_mask]) * self._camera.depth_scale * 100
            max_d = np.max(self._last_depth[valid_mask]) * self._camera.depth_scale * 100
            cv2.putText(depth_display, f"Range: {min_d:.0f}-{max_d:.0f}cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return depth_display
    
    def _print_block_info(self, block: Block):
        """블록 정보 출력"""
        print("\n" + "=" * 50)
        print(f"🎯 블록 정보 #{block.click_order}")
        print("=" * 50)
        print(f"  📍 중심점 (픽셀): {block.center_2d}")
        
        if block.is_valid:
            pos = block.center_3d_mm
            print(f"  📍 3D 좌표: X={pos[0]:.1f}mm, Y={pos[1]:.1f}mm, Z={pos[2]:.1f}mm")
            print(f"  📏 한 변 길이: {block.side_length_mm:.1f} mm")
            print(f"  📐 각도: {block.angle:.1f}°")
            print(f"  📊 면적: {block.area:.1f} px²")
        else:
            print("  ⚠️ 깊이 정보 없음")
        
        print("=" * 50 + "\n")


# ===== 전역 함수들 =====

def stop_system():
    """카메라 및 시스템 종료"""
    global _system

    if _system is not None:
        _system.stop()
        _system = None


def get_clicked_blocks():
    """저장된 블록 리스트 반환"""
    if _system is None:
        return []
    return _system.get_clicked_blocks()


def get_clicked_floor_points():
    """저장된 바닥 포인트 리스트 반환"""
    if _system is None:
        return []
    return _system.get_clicked_floor_points()


def get_block_summaries():
    """클릭된 블록 요약 반환"""
    if _system is None:
        raise RuntimeError("System not started")

    summaries = []
    for b in _system.get_clicked_blocks():
        if b.center_3d is None:
            continue

        summaries.append({
            "center_3d": tuple(float(x) for x in b.center_3d),
            "angle": float(b.angle),
            "real_width_mm": float(b.real_width_mm),
            "click_order": int(b.click_order)
        })

    return summaries


def get_floor_summaries():
    """클릭된 바닥 포인트 요약 반환"""
    global _system
    if _system is None:
        raise RuntimeError("System not started")

    summaries = []
    for f in _system._clicked_floor_points:
        summaries.append({
            "pixel": tuple(f.pixel),
            "depth": float(f.depth),
            "point_3d": tuple(float(x) for x in f.point_3d) if f.point_3d else None,
            "timestamp": f.timestamp
        })
    return summaries


def run_gui():
    """디버그 GUI 실행"""
    if _system is None:
        raise RuntimeError("System not started. Call start_system() first.")

    _system.run_debug()


def start_system():
    """BlockDetectionSystem 시작"""
    global _system

    if _system is None:
        _system = BlockDetectionSystem()
        _system.start()

    return _system