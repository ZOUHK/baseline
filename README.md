1. inference.py 是推理的入口，inference.sh 则对此做了参数的封装。python 的执行需要严格仿照 inference.sh 中所给出的范式。
2. api_list.json 是全部的API工具，已转成标准的API格式，其中 "paths" 字段用于访问 API。
3. model 文件是本次大赛的参考 retrieval model，用于检索用户 query 对应的 APIs。
4. dataset.json 是本次的赛题数据集，选手们可以基于赛题和api_list，构建自己的数据集训练一个 retrieval model。
