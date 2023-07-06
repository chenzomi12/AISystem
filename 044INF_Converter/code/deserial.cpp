#include <fstream>
 #include <iostream>
 #include <vector>
 ​
 #include "net_generated.h"
 using namespace PiNet;
 ​
 int main() {
     std::ifstream infile;
     infile.open("net.mnn", std::ios::binary | std::ios::in);
     infile.seekg(0, std::ios::end);
     int length = infile.tellg();
     infile.seekg(0, std::ios::beg);
     char* buffer_pointer = new char[length];
     infile.read(buffer_pointer, length);
     infile.close();
 ​
     auto net = GetNet(buffer_pointer);
 ​
     auto ConvOp = net->oplists()->Get(0);
     auto ConvOpT = ConvOp->UnPack();
 ​
     auto PoolOp = net->oplists()->Get(1);
     auto PoolOpT = PoolOp->UnPack();
 ​
     auto inputIndexes = ConvOpT->inputIndexes;
     auto outputIndexes = ConvOpT->outputIndexes;
     auto type = ConvOpT->type;
     std::cout << "inputIndexes: " << inputIndexes[0] << std::endl;
     std::cout << "outputIndexes: " << outputIndexes[0] << std::endl;
 ​
     PiNet::OpParameterUnion OpParameterUnion = ConvOpT->parameter;
     switch (OpParameterUnion.type) {
         case OpParameter_Conv: {
             auto ConvOpParameterUnion = OpParameterUnion.AsConv();
             auto k = ConvOpParameterUnion->kernelX;
             std::cout << "ConvOpParameterUnion, k: " << k << std::endl;
             break;
         }
         case OpParameter_Pool: {
             auto PoolOpParameterUnion = OpParameterUnion.AsPool();
             auto k = PoolOpParameterUnion->padX;
             std::cout << "PoolOpParameterUnion, k: " << k << std::endl;
             break;
         }
         default:
             break;
     }
     return 0;
 }