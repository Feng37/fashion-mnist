//
//  randomSelect.cpp
//  test-eigen
//
//  Created by gufeng on 2018-04-28.
//  Copyright © 2018 gufeng. All rights reserved.
//

#include "randomSelect.hpp"

vector<string> randomSelect(string sourceFileName, int n) {
  filebuf fb;
  vector<string> selectedData;
  if (fb.open(sourceFileName, ios::in)) {
    istream file(&fb);
    string curLine;
    int i = 0;
    while (getline(file, curLine)) {
      if (i < n) {
        selectedData.push_back(curLine);
      }
      else {
        int randomNum = rand() % (i + 1);
        if (randomNum < n) {
          selectedData[randomNum] = curLine;
        }
      }
      i += 1;
    }
  }
  else {
    cout << "No such input file." << endl;
  }
  fb.close();
  return selectedData;
}
//import random
//def sample(filename, n = 5):
//assert(filename is not None)
//result = []
//with open(filename, mode = "r") as f:
//for i in range(n):
//line = f.readline()
//if line == '':
//return result
//else:
//result.append(line.strip())
//i = n
//for line in f:
//k = random.randint(0, i)
//if k < n:
//result[k] = line.strip()
//i += 1
//return result
