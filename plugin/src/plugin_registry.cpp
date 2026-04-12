#include "bevpool_plugin.h"
#include "alignbev_plugin.h"
#include "gatherbev_plugin.h"

#include <NvInferRuntime.h>

namespace tio {

REGISTER_TENSORRT_PLUGIN(BevPoolPluginCreator);
REGISTER_TENSORRT_PLUGIN(AlignBevPluginCreator);
REGISTER_TENSORRT_PLUGIN(GatherBevPluginCreator);

}  // namespace tio
