import os
import xml.etree.cElementTree as et

file_dir = r'C:\PAPI2020\training_data'

for dir_path, subpaths, files in os.walk(file_dir, False):
    for f in files:
        file_xml = os.path.join(dir_path, f)
        filename, file_extension = os.path.splitext(file_xml)
        if file_extension.upper() not in ['.XML']:
            continue

        tree = et.parse(file_xml)
        root = tree.getroot()
        print(root.tag, ":", root.attrib)  # 打印根元素的tag和属性
        print('root ok')

        datas_node = root.find(".//Regions")
        for children in datas_node:
            print(children.tag, ":", children.attrib)

        exit(0)

        # 遍历xml文档的第二层
        for child in root:
            print(child.tag, ":", child.attrib)
            for children in child:
                print(children.tag, ":", children.attrib)
        print('ok')
        exit(0)

        print(file_xml)


print('OK')

