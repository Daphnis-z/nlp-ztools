# nlp-ztools
本项目包含几种常用 NLP算法的实现：关键词(keyword)、命名实体(named entity)、自动摘要(abstract)、文本相似度比较(text similarity)等。

本项目最大的特点是以**工程化**的思维对算法进行了**改进**和**封装**，真正做到**“开箱即用”**。另外，本项目基于 python3，依赖 jieba,tensorflow等第三方库。

**各个算法模块简介**：

- **关键词**

  在 jieba的基础上，进行了一些封装。可以很方便的在 etc/user_words.dict中添加**用户词典**，以加强对一些领域特有关键词的识别。

  调用举例：

  ```python
  kw_extract = KeywordExtraction(stopword_file='etc/stopwords.txt', keyword_weight=0.25)
  content = file_util.read_whole_file('data/test001.txt')
  keyword_list = kw_extract.extract_keyword(content)
  ```

  输出：

  ```
  extract keywords: ['楼市', '分化', '住房', '房子', '房价', '孙宏斌', '城市', '行业', '局部楼市']

- **命名实体**

  利用训练好的模型文件，对 人名、地名和组织机构名进行识别。模型文件存放在 data目录，可以直接使用。

  调用举例：

  ```python
  content = file_util.read_whole_file('data/test001.txt')
  entities = named_entity.extract_entity(content)
  ```

  输出：

  ```
  extract named entities: ['成都', '土地管理局', '孙宏斌']

- **自动摘要**

  自动摘要是基于 TextRank算法思想，提取文本中比较重要的语句作为摘要。

  调用举例：

  ```python
  content = file_util.read_whole_file('data/test005.txt')
  abstract = AutoAbstract().generate_abstract(content, 3)
  ```

  输出：

  ```
  generate abstract: 
    坚持以人民为中心的价值导向为我国科技创新提供价值导航，就能推动我国科技事业日益强大而美好，更好地造福于人民，造福于世界
  科技创新的目的，赋予创新生机与活力，科技创新的目的决定创新的价值取向或导向
  加快科技创新，离不开正确价值导向

- **文本相似度**

  文本相似度指的是计算两篇文本的相似度，基于 MinHash算法思想。

  调用举例：

  ```python
  text1 = file_util.read_whole_file('data/test001.txt')
  text2 = file_util.read_whole_file('data/test002.txt')
  similarity = text_similarity.calc_similarity(text1, text2)
  ```

  输出：

  ```
  data/test001.txt and data/test002.txt similarity: 0.94

