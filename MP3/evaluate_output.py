import os
import shutil
import subprocess
import tempfile
from typing import Tuple

def _run(cmd: list, cwd: str, timeout: int = 20) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err

def evaluate_java_response(main_src) -> Tuple[bool, str]:
    """
    Returns (is_correct, full_log)
    - is_correct True iff: compiles and 'java Main' exits with code 0 (no AssertionError)
    - full_log contains compilation and runtime stderr/stdout to help debugging
    """
    log = []

    # Create a temporary directory for compilation/execution (auto-cleaned via shutil.rmtree)
    tempdir = tempfile.mkdtemp(prefix="mp3_eval_")
    try:
        # Write the combined Main.java file (test + solution) to the temp directory
        main_path = os.path.join(tempdir, "Main.java")
        with open(main_path, "w") as f:
            f.write(main_src)
            
        # ===== COMPILATION PHASE =====
        # Run javac to compile Main.java in the temp directory
        rc, out, err = _run(["javac", "Main.java"], cwd=tempdir)
        
        # Log compilation output for debugging
        log.append("=== compilation javac stderr ===")
        log.append(err if err else "(empty)")
        log.append("")  # blank line
        log.append("=== compilation javac stdout ===")
        log.append(out if out else "(empty)")
        log.append("")  # blank line
        
        # Check if compilation failed (non-zero return code)
        if rc != 0:
            log.append("Compilation failed.")
            return False, "\n".join(log)  # Return early with failure

        # ===== EXECUTION PHASE =====
        # Run the compiled Java program (executes Main.main())
        rc, out, err = _run(["java", "Main"], cwd=tempdir)
        
        # Log runtime output for debugging
        log.append("=== execution java stdout ===")
        log.append(out if out else "(empty)")
        log.append("")  # blank line
        log.append("=== execution java stderr ===")
        log.append(err if err else "(empty)")
        log.append("")  # blank line
        
        # Check if runtime failed (non-zero exit code indicates assertion failure or exception)
        if rc != 0:
            log.append(f"Execution failed:\n {rc}.")
            return False, "\n".join(log)  # Return early with failure

        # ===== SUCCESS =====
        # Both compilation and execution succeeded (rc == 0 means all assertions passed)
        log.append("All tests passed.")
        return True, "\n".join(log)

    finally:
        # Clean up: delete the temporary directory and all its contents
        # ignore_errors=True ensures cleanup doesn't fail if dir is already gone
        shutil.rmtree(tempdir, ignore_errors=True)  