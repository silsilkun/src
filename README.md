# src
1. 완성본: main.py, camera.py
2. 실행은 "main.py"만
3. 실행 순서
   
    3-1. 터미널에서 ~/Jina 워크 스페이스로 이동 

    3-2. colcon build & /setup.bash 필수 

    3-3. 터미널 두 개에서 각각 ros2 launch bringup, ros2 run 실행

      * ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=110.120.1.18 port:=12345 model:=e0509
      
      * ros2 run tower_builder stb
