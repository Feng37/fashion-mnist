//
//  main.cpp
//  test-eigen
//
//  Created by gufeng on 2018-04-28.
//  Copyright © 2018 gufeng. All rights reserved.
//


#include "randomSelect.hpp"


int main() {
  string sourceFileName = "testData.txt";
  vector<string> selectedData;
  selectedData = randomSelect(sourceFileName, n);
  for (int i = 0; i < selectedData.size(); i++) {
    cout << selectedData[i] << endl;
  }
  return 0;
}

