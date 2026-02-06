需要实现函数1: 输入一段文本和本地文件路径，提取该文本在该文件的第几行到第几行, 实现方式为提取文本第一行，查看第一次匹配的行号，以及最后一行内容，查看匹配中最后一次匹配的行号，这样即使有多个匹配，确保包含最大的信息量。

函数2：根据文件路径和第一行和最后一行行号，返回对应内容

函数3实现 找到 chunk 所在位置 → 扩展上下文： 利用函数1和函数2，如果设置的是上下扩展行号的比例，用这段文本的start和end，end-start*比例，得到上下扩展的行号，然后调用函数2返回扩展后的内容 如果设置的是提取文本所在markdown块，需要具备markdown解析能力，拿到该段文本所在的markdown完整信息块
返回格式为：{file_path: str, start_line: int, end_line: int, content: str}

函数4实现 根据关键字本地搜索：输入[string] 查找每个关键字在哪些文件以及对应的行号，
返回格式为：
{keyword :[
  {file_path: str, start_line: int, end_line: int, content: str}
  {file_path: str, start_line: int, end_line: int, content: str}
]
}一个关键字可以匹配多个文件，同样这里返回的行数也有两种参数，一种是固定上下扩展n行，一种是找到该段文本所在的markdown完整信息块