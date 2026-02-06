| 参数                     | 默认/常用值                                    | 作用                      | 建议                                        |
| ---------------------- | ----------------------------------------- | ----------------------- | ----------------------------------------- |
| `hnsw_m`               | 16（你贴的）                                   | 每个节点的双向连接数，影响索引稠密度和搜索精度 | 16~64，数据量大可以适当调大，提高召回率，但索引构建慢，内存占用增加      |
| `hnsw_ef_construction` | 64                                        | 构建索引时候的候选列表大小，影响索引质量    | 100~200 对大数据集更稳，64 对小数据量够用                |
| `hnsw_ef_search`       | 40                                        | 搜索时候的候选列表大小，影响召回率和速度    | 10~200 可调，值越大召回率越高，搜索越慢                   |
| `hnsw_dist_method`     | `"vector_cosine_ops"` 或 `"vector_l2_ops"` | 距离计算方式：cosine 或 L2      | 文本 embedding 常用 cosine；图像 embedding 可用 L2 |


## 表结构：
id | bigint | | not null | nextval('data_askany_faq_vectors_id_seq'::regclass)

text | character varying | | not null |

metadata_ | json | | |

node_id | character varying | | |

embedding | vector(1024) | | |

Indexes:

"data_askany_faq_vectors_pkey" PRIMARY KEY, btree (id)

"askany_faq_vectors_idx_1" btree ((metadata_ ->> 'ref_doc_id'::text))

"data_askany_faq_vectors_embedding_idx" hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')