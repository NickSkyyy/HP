import subprocess as sp

import os
os.environ['PATH'] += "D:\\Anaconda\\envs\\graph-generation"
# print(os.environ["PATH"])
# input()


if __name__ == "__main__":
  print("here")
  output = None
  try:
    output = sp.check_output(['.\\orca.exe', 'node', '4', '.\\tmp.txt', 'std'], stderr=sp.STDOUT)
  # output = sp.check_output(["F:\\project\\PRO3\\eval\\orca\\orca.exe", "node", "4", "F:\\project\\PRO3\\eval\\orca\\tmp.txt", "std"])
  # output = sp.check_output(["python", "hello.py"])
  # output = sp.run(['.\\orca.exe', 'node', '4', '.\\tmp.txt', 'std'])
    output = output.decode('utf8').strip()
  except sp.CalledProcessError as e:
    print(e.returncode)
    print(e.output.decode())
  print(output)
  print("done")