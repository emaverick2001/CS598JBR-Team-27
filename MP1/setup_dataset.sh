pip3 install datasets==2.16.1
git clone https://github.com/openai/human-eval
pip3 install -e human-eval

# 1) Uncomment a specific exec(...) line in execution.py (remove leading `#` but keep indentation)
sed -i 's/^#\([[:space:]]*\)exec(check_program, exec_globals)/\1exec(check_program, exec_globals)/' human-eval/human_eval/execution.py

# 2) Comment out (prefix with '#') a specific assert line in evaluation.py
sed -i '/assert len(completion_id) == len(problems), "Some problems are not attempted."/s/^/#/' human-eval/human_eval/evaluation.py

pip3 install jsonlines
