#include <fstream>
 #include <iostream>
 ​
 #include "net_generated.h"
 using namespace PiNet;
 ​
 int main() {
     flatbuffers::FlatBufferBuilder builder(1024);
 ​
     // table ConvT
     auto ConvT = new PiNet::ConvT;
     ConvT->kernelX = 3;
     ConvT->kernelY = 3;
     // union ConvUnionOpParameter
     OpParameterUnion ConvUnionOpParameter;
     ConvUnionOpParameter.type = OpParameter_Conv;
     ConvUnionOpParameter.value = ConvT;
     // table OpT
     auto ConvTableOpt = new PiNet::OpT;
     ConvTableOpt->name = "Conv";
     ConvTableOpt->inputIndexes = {0};
     ConvTableOpt->outputIndexes = {1};
     ConvTableOpt->type = OpType_Conv;
     ConvTableOpt->parameter = ConvUnionOpParameter;
 ​
     // table PoolT
     auto PoolT = new PiNet::PoolT;
     PoolT->padX = 3;
     PoolT->padY = 3;
     // union OpParameterUnion
     OpParameterUnion PoolUnionOpParameter;
     PoolUnionOpParameter.type = OpParameter_Pool;
     PoolUnionOpParameter.value = PoolT;
     // table Opt
     auto PoolTableOpt = new PiNet::OpT;
     PoolTableOpt->name = "Pool";
     PoolTableOpt->inputIndexes = {1};
     PoolTableOpt->outputIndexes = {2};
     PoolTableOpt->type = OpType_Pool;
     PoolTableOpt->parameter = PoolUnionOpParameter;
 ​
     // table NetT
     auto netT = new PiNet::NetT;
     netT->oplists.emplace_back(ConvTableOpt);
     netT->oplists.emplace_back(PoolTableOpt);
     netT->tensorName = {"conv_in", "conv_out", "pool_out"};
     netT->outputName = {"pool_out"};
     // table Net
     auto net = CreateNet(builder, netT);
     builder.Finish(net);
 ​
     // This must be called after `Finish()`.
     uint8_t* buf = builder.GetBufferPointer();
     int size = builder.GetSize();  // Returns the size of the buffer that
                                    //`GetBufferPointer()` points to.
     std::ofstream output("net.mnn", std::ofstream::binary);
     output.write((const char*)buf, size);
 ​
     return 0;
 }