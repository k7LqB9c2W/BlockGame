<img width="1919" height="992" alt="image" src="https://github.com/user-attachments/assets/0a21b184-a01f-4011-9130-0ac2cd4deaaf" />


Build Steps

  2. Visual Studio generator (lets you keep using -A x64)

     cmake -S . -B build_vs -G "Visual Studio 17 2022" -A x64
     cmake --build build_vs --config Release   # or Debug
