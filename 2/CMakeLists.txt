add_executable (2 "main.cpp"      )

set_property(TARGET 2 PROPERTY CXX_STANDARD 17)
set_property(TARGET 2 PROPERTY CXX_STANDARD_REQUIRED ON)


if (UNIX)
	target_compile_options(2 PRIVATE -mavx2 -mfma -O3 -march=native)
elseif (MSVC)
	target_compile_options(2 PRIVATE $<$<CONFIG:Release>:/O2 /arch:avx2>)
    target_compile_options(2 PRIVATE $<$<CONFIG:Debug>:/RTC1>)
endif()

add_custom_command(
    TARGET 2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/data
            $<TARGET_FILE_DIR:2>/data
    COMMENT "Copying runtime data files..."
)