README



## 各文件功能

​	**api.py**：暴露api接口

​	**face_img.py**：人脸检测468关键点，最后加了两个函数方便单独调用

​	**llm.py**：运行大模型输出分析报告

​	**llm_config.py**：具体的llm信息，目前使用的是我个人的deepseek接口

​	**settings.py**：全局配置，但是llm上的优先级低于llm_config.py

​	**pubilc**：存储468点检测后的图片

​	**NotoSansCJKsc-Regular.otf**：中文字体，在图片上使用汉字

​	**id_ops.py**和**meshid_store.py**：尝试加入的云端关键点数据库功能，测试效果较差



### 字段含义

​	**image_url**：前端上传的图片url

​	**with_ai**：是否开启大模型分析功能，默认为true

​	**metrics**：关键点信息，无image_url上传时通过关键点信息返回大模型分析结果；存在image_url时无效

​	**job_id**：云端收到图片后创建的任务ID



### <u>**代码逻辑**</u>

**有图情况**

​	发送 **image_url + with_ai** ：云端收到字段后立刻走 **face_img.py** 返回468关键点识别的图片并返回给前端，同时创建 **job_id** ，将468关键点信息发送给 **llm.py** 调用大模型在后台进行分析。前端在收到关键点图片后可以查询大模型分析结果，即发送对 **job_id** 的查询到云端，云端若还未结束查询则前端需要等待，调用大模型大概需要一分钟。大模型分析结果默认保存一小时，一小时内在只要发送 **job_id** 都可以查询到。

分离检测和大模型的原因：目前的大模型是直接调的 deepseek 的接口，速度很慢，目前的逻辑可以直接输出关键点检测的图片，不需要等大模型。

**无图情况**

​	发送 **metrics + with_ai** ：云端收到字段后不再走 **face_img.py**，而是直接创建 **job_id** 并将 **metrics** 发送给 **llm.py **调用大模型在后台进行分析。其余同理。



### curl测试示例，格式及返回如下

**有图情况**

>   (face) mahaiyan@beta88:/mnt1/mhy/projects/MediaPipe_Face_Mesh/facemesh$ curl -X POST "http://127.0.0.1:8000/analyze" \
>
>   -H "Content-Type: application/json" \
>   -d '{
>    "image_url": "https://c-ssl.dtstatic.com/uploads/blog/202309/08/8gSMABJACgXEdZM.thumb.1000_0.png",
>    "with_ai": true,
>    "return_data_url": false
>   }'
>
>   {"result_url":"http://127.0.0.1:8000/files/064e76bb8c5f487793874c605beafa36.jpg","filename":"064e76bb8c5f487793874c605beafa36.jpg","ai_job":{"job_id":"03969868b3194f74a86db8f96093cd6b","status":"running","created_at":"2025-09-04T09:58:10.250918+00:00","expires_at":"2025-09-04T10:58:10.250939+00:00"}}
>
>   (face) mahaiyan@beta88:/mnt1/mhy/projects/MediaPipe_Face_Mesh/facemesh$ curl "http://127.0.0.1:8000/jobs/03969868b3194f74a86db8f96093cd6b?wait=1&timeout=55"
>
>   {"job_id":"03969868b3194f74a86db8f96093cd6b","status":"completed","result_url":"http://127.0.0.1:8000/files/064e76bb8c5f487793874c605beafa36.jpg","filename":"064e76bb8c5f487793874c605beafa36.jpg","face_analysis":{"twelve_palaces":[{"palace_name":"1. 命宫","description":"命宫区域宽度适中但高度较低，饱满度略低，可能表示当前精力状态较为平稳，建议保持规律作息以维持活力。"},{"palace_name":"2. 兄弟宫","description":"左右兄弟宫面积较大且宽度均衡，饱满度接近中性，暗示人际关系可能较为和谐，适合多与朋友交流，享受社交生活。"},{"palace_name":"3. 夫妻宫","description":"夫妻宫左右宽度相似，饱满度轻微波动，显示情感关系可能处于稳定期，注重日常沟通能增进亲密感。"},{"palace_name":"4. 子女宫","description":"子女宫高度较低，饱满度接近中性，可能反映对家庭或后代的关注度适中，建议平衡个人与家庭时间。"},{"palace_name":"5. 财帛宫","description":"财帛宫宽度适中但饱满度略低，提示财务状态可能需谨慎管理，养成储蓄习惯有助于稳定经济。"},{"palace_name":"6. 疾厄宫","description":"疾厄宫高度较高但宽度较窄，饱满度中性，表示身体健康状况总体良好，注意适度运动和休息以预防小问题。"},{"palace_name":"7. 迁移宫","description":"左右迁移宫面积和饱满度有差异，左宫更饱满，可能暗示出行或变化机会较多，保持开放心态迎接新体验。"},{"palace_name":"8. 奴仆宫","description":"左右奴仆宫不对称，左宫面积较大且饱满度较高，显示在合作或服务关系中可能更主动，建议平等对待他人以维持和谐。"},{"palace_name":"9. 官禄宫","description":"官禄宫面积很小，饱满度较低，可能表示事业或学业进展平缓，专注于小目标逐步积累成就。"},{"palace_name":"10. 田宅宫","description":"左右田宅宫面积差异大，右宫更饱满，提示对家庭或环境的舒适度感受较强，营造温馨空间能提升幸福感。"},{"palace_name":"11. 福德宫","description":"福德宫面积很小，饱满度较低，可能反映内心满足感需加强，通过 hobbies 或冥想培养平和心态。"},{"palace_name":"12. 父母宫","description":"左右父母宫面积和饱满度相近且略低，表示与长辈关系可能平稳，多沟通能增进理解和支持。"}],"three_court_five_eye":{"three_court":{"description":"三庭比例显示上庭略短，中庭和下庭较均衡，整体面部结构协调，可能性格稳重，适合保持平衡的生活方式。"},"five_eye":{"description":"五眼比例中，中间部分较宽，两侧略窄，暗示视觉焦点集中，有助于专注力，但建议多角度观察事物以避免局限。"}},"ai_destiny":"基于面部几何特征分析，整体面部结构较为均衡，三庭比例协调，五眼分布显示中间区域较突出，可能意味着你在生活中注重核心事务，拥有稳定的基础。兄弟宫和夫妻宫的饱满度提示人际关系和谐，适合维护社交网络；财帛宫和官禄宫的略低饱满度建议在财务和事业上采取稳健策略，逐步积累。迁移宫的不对称性表明变化机会存在，保持灵活性可丰富生活体验。健康方面，疾厄宫的状态良好，但需注意日常保健。总体而言，这些特征反映了平和与潜力的结合，鼓励你以积极心态面对日常，通过小调整提升整体 well-being，避免过度解读或压力。记住，面部特征只是生活的一部分，真正的命运由行动和选择塑造。"}}

**无图情况，只传关键点信息**

>   (face) mahaiyan@beta88:/mnt1/mhy/projects/MediaPipe_Face_Mesh/facemesh$ curl -X POST "http://127.0.0.1:8000/analyze" \
>
>   -H "Content-Type: application/json" \
>   -d '{
>   "with_ai": true,
>   "metrics": {
>     "three": {
>       "upper_ratio": 0.30,
>       "middle_ratio": 0.35,
>       "lower_ratio": 0.35,
>       "values_px": [120, 140, 150],
>       "anchors": {"forehead": 10, "brow": 9, "nose_base": 2, "chin": 152}
>     },
>     "five": {
>       "eye_widths_px": [80, 40, 42, 82, 90],
>       "eye_ratios": [0.16, 0.16, 0.17, 0.17, 0.34],
>       "description": "五眼比例: 16.0%, 16.0%, 17.0%, 17.0%, 34.0%",
>       "anchors": {"L_border": 234, "L_outer": 130, "L_inner": 133, "R_inner": 362, "R_outer": 359, "R_border": 454}
>     },
>     "palaces": {
>       "父母宫_左":  {"area": 1200.5, "width": 45.3, "height": 30.1, "fullness_z_mean": -0.02, "count": 18, "indices": [10,11,12],   "box": [100, 50, 145, 80]},
>       "父母宫_右":  {"area": 1180.2, "width": 44.8, "height": 29.7, "fullness_z_mean": -0.01, "count": 17, "indices": [20,21,22],   "box": [150, 50, 195, 80]},
>       "官禄宫":    {"area": 800.0,  "width": 30.0, "height": 25.0, "fullness_z_mean": 0.00,  "count": 12, "indices": [30,31,32],   "box": [120, 40, 150, 65]},
>       "福德宫":    {"area": 860.0,  "width": 34.0, "height": 26.0, "fullness_z_mean": 0.01,  "count": 13, "indices": [33,34,35],   "box": [122, 60, 156, 86]},
>       "田宅宫_左":  {"area": 760.0,  "width": 34.0, "height": 25.0, "fullness_z_mean": -0.03, "count": 14, "indices": [50,51,52],   "box": [ 90, 65, 124, 90]},
>       "田宅宫_右":  {"area": 740.0,  "width": 33.0, "height": 25.0, "fullness_z_mean": -0.02, "count": 13, "indices": [53,54,55],   "box": [160, 65, 193, 90]},
>       "命宫":      {"area": 610.0,  "width": 28.0, "height": 24.0, "fullness_z_mean": 0.00,  "count": 12, "indices": [60,61,62],   "box": [118, 85, 146,110]},
>       "兄弟宫_左":  {"area": 720.0,  "width": 30.0, "height": 26.0, "fullness_z_mean": -0.01, "count": 13, "indices": [70,71,72],   "box": [ 85, 70, 115, 96]},
>       "兄弟宫_右":  {"area": 710.0,  "width": 31.0, "height": 26.0, "fullness_z_mean": -0.02, "count": 13, "indices": [73,74,75],   "box": [165, 70, 195, 96]},
>       "子女宫":    {"area": 650.0,  "width": 40.0, "height": 20.0, "fullness_z_mean": 0.01,  "count": 11, "indices": [80,81,82],   "box": [120,120, 160,140]},
>       "夫妻宫_左":  {"area": 780.0,  "width": 36.0, "height": 28.0, "fullness_z_mean": 0.02,  "count": 14, "indices": [90,91,92],   "box": [ 80,110, 116,138]},
>       "夫妻宫_右":  {"area": 770.0,  "width": 35.0, "height": 28.0, "fullness_z_mean": 0.01,  "count": 14, "indices": [93,94,95],   "box": [165,110, 200,138]},
>       "财帛宫":    {"area": 900.0,  "width": 40.0, "height": 30.0, "fullness_z_mean": 0.03,  "count": 17, "indices": [100,101,102], "box": [125,140, 165,170]},
>       "疾厄宫":    {"area": 500.0,  "width": 25.0, "height": 20.0, "fullness_z_mean": -0.01, "count":  9, "indices": [110,111,112], "box": [130,160, 155,180]},
>       "迁移宫_左":  {"area": 400.0,  "width": 20.0, "height": 18.0, "fullness_z_mean": 0.00,  "count":  8, "indices": [120,121,122], "box": [ 60, 80,  80, 98]},
>       "迁移宫_右":  {"area": 390.0,  "width": 20.0, "height": 18.0, "fullness_z_mean": 0.00,  "count":  8, "indices": [123,124,125], "box": [200, 80, 220, 98]},
>       "仆役宫_左":  {"area": 640.0,  "width": 28.0, "height": 28.0, "fullness_z_mean": -0.02, "count": 12, "indices": [130,131,132], "box": [ 85,200, 113,228]},
>       "仆役宫_右":  {"area": 630.0,  "width": 28.0, "height": 28.0, "fullness_z_mean": -0.01, "count": 12, "indices": [133,134,135], "box": [167,200, 195,228]}
>     }
>   }
>   }'
>
>   {"result_url":"http://127.0.0.1:8000/files/2e2a808c07214395a29233db783d4f02.jpg","filename":"2e2a808c07214395a29233db783d4f02.jpg","ai_job":{"job_id":"9f476924cab345ccbed9846727942bc6","status":"running","created_at":"2025-09-02T08:57:48.317211+00:00","expires_at":"2025-09-02T09:57:48.317225+00:00"}}
>
>   (face) mahaiyan@beta88:/mnt1/mhy/projects/MediaPipe_Face_Mesh/facemesh$ curl "http://127.0.0.1:8000/jobs/9f476924cab345ccbed9846727942bc6?wait=1&timeout=55"
>
>   {"job_id":"9f476924cab345ccbed9846727942bc6","status":"completed","result_url":"http://127.0.0.1:8000/files/2e2a808c07214395a29233db783d4f02.jpg","filename":"2e2a808c07214395a29233db783d4f02.jpg","face_analysis":{"twelve_palaces":[{"palace_name":"1. 命宫","description":"命宫区域面积适中，饱满度均衡，高度和宽度比例协调。这通常表示当前生活状态较为稳定，适合保持日常的平衡与和谐。"},{"palace_name":"2. 兄弟宫","description":"左右兄弟宫面积相近，饱满度略低，但整体对称。可能反映与兄弟姐妹或朋友的关系需要一些维护，建议多沟通以增进情感。"},{"palace_name":"3. 夫妻宫","description":"夫妻宫面积较大，饱满度略高，左右对称。这暗示人际关系或伴侣互动可能较为积极，鼓励保持开放和包容的态度。"},{"palace_name":"4. 子女宫","description":"子女宫面积较小，但饱满度均衡。可能表示在家庭或创意方面需要更多投入，建议花时间培养兴趣或关怀他人。"},{"palace_name":"5. 财帛宫","description":"财帛宫面积较大，饱满度较高。这通常与资源管理相关，提示当前可能有较好的财务意识，鼓励保持理性和规划。"},{"palace_name":"6. 疾厄宫","description":"疾厄宫面积较小，饱满度略低。提醒注意日常健康习惯，如规律作息和适度运动，以维持整体 well-being。"},{"palace_name":"7. 迁移宫","description":"左右迁移宫面积小，饱满度均衡。可能表示外出或变化的机会较少，建议偶尔尝试新事物来丰富生活体验。"},{"palace_name":"8. 奴仆宫","description":"奴仆宫面积适中，饱满度略低，但对称。反映在社交或团队合作中可能需更多主动，鼓励建立互信关系。"},{"palace_name":"9. 官禄宫","description":"官禄宫面积较小，饱满度均衡。提示事业或责任方面可能处于平稳期，适合专注于细节和持续努力。"},{"palace_name":"10. 田宅宫","description":"左右田宅宫面积相近，饱满度略低。可能表示家庭或环境方面需要一些关注，建议营造舒适的生活空间。"},{"palace_name":"11. 福德宫","description":"福德宫面积较大，饱满度略高。这常与内在幸福感相关，鼓励多参与愉悦活动，培养 positive mindset。"},{"palace_name":"12. 父母宫","description":"左右父母宫面积大且对称，饱满度均衡。可能反映家庭支持或传统价值观较强，建议珍惜亲情联系。"}],"three_court_five_eye":{"three_court":{"description":"三庭比例显示上庭略短，中庭和下庭均衡，整体面部高度分布较为和谐，暗示生活节奏可能偏向务实和稳定。"},"five_eye":{"description":"五眼比例中，眼睛区域宽度分布均匀，但右眼外侧稍宽，整体对称性良好，表示视觉感知和社交互动可能较为平衡。"}},"ai_destiny":"基于面部几何特征的综合分析，您的面部结构整体表现出良好的对称性和均衡性，三庭比例中上庭略短但中下庭协调，五眼比例均匀，暗示生活态度偏向稳定和务实。各宫位中，父母宫和财帛宫面积较大且饱满，可能反映较强的家庭支持和资源管理能力；夫妻宫和福德宫饱满度略高，提示人际关系和内在幸福感较为积极；而疾厄宫和迁移宫面积较小，提醒注意健康维护和偶尔尝试新体验。整体而言，这并非命运预言，而是基于几何指标的温和解读，鼓励您保持当前的生活平衡，多关注健康、加强社交互动，并享受日常中的小确幸。记住，面部特征只是外在表现，真正的幸福源于内心的选择和行动。"}}