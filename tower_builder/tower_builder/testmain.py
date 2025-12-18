import rclpy
import DR_init
import sys
from vtestDataS import vtestData


def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "e0509"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    VEL = 50
    ACC = 50

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import(
        movej, movel, posj, 
        set_robot_mode, ROBOT_MODE_AUTONOMOUS
    ) 

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    block, floor = vtestData()

    rx, ry, rz = 90, 180, 90
    #=================================
    # 쌓기
    #=================================
    for i in range(len(block)):
        camx, camy, camz = block[i]["center_3d"]
        


    P0 = posj(0,0,90,0,90,0)
    print("홈 위치로 이동합니다")
    movej(P0, VEL, ACC)

    print("1. movej 예제를 실행합니다.")
    movej([0, 0, 90, -90, 90, 0], VEL, ACC)
    # movej([90, 0, 90, 0, 90, 0], VEL, ACC)
    # movej([0, 0, 60, 120, 90, 0], VEL, ACC)
    
    print("홈 위치로 이동합니다")
    movej(P0, VEL, ACC)

    print("2. movel 예제를 실행합니다.")
    movel([373, 100, 404, 12, -180, 12], VEL, ACC)
    movel([473, 100, 404, 12, -180, 12], VEL, ACC)
    movel([373, 100, 504, 12, -180, 12], VEL, ACC)

    print("홈 위치로 이동합니다")
    movej(P0, VEL, ACC)

    print("예제 프로그램이 종료되었습니다.")
    rclpy.shutdown()

if __name__ == '__main__':
    main()