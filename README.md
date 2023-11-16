# DesignAuto

本项目旨在建立一个库共享、规范化的统一结构设计软件。

## 安装 

[依赖&安装](https://u2c6hvagv9.feishu.cn/docx/doxcnj4aGvQGuscrg8t8X36qQsc)

## 项目结构  

本项目的结构按层级依次分为 

### 3rd  

[3rd 添加指南](https://u2c6hvagv9.feishu.cn/docx/doxcnjyJbKZhCdKPN4i9eqXL7Mb)

第三方依赖库，目前通用的算法库为 CGAL 和 libigl，这两个库和版本依赖较强，可能会随着其版本更迭改变 api，另外他们都是 header-only 库，即可只有头文件存在，因此
这里把他们随项目放入 3rd 中。像 eigen、 spdlog 之类的库，因为版本比较稳定，并且安装方便，所以建议直接安装在操作系统内。

值得注意的是 3rd 并不处于项目的 git 管理中，而是作为子模块存在，真实地址为 [GeometryMain/designauto-3rd](http://118.195.195.192:3000/GeometryMain/designauto-3rd)，
因此 clone 项目的时候要记得 `git clone --recursive $(project_path)`

### assets

[assets 管理指南](https://u2c6hvagv9.feishu.cn/docx/doxcnHUcaaMQu8P44fvgtDUv5Rd)

整个项目都有可能依赖的静态资源库，使用 git lfs 进行管理。

### da-cpt [ComPuTation]

[cpt 开发指南](https://u2c6hvagv9.feishu.cn/docx/doxcn3EkVTVIjMHTuqTfc72dCYe)

与业务无关的高度抽象算法模块。不依赖于除数学库（如 Eigen、ipopt等）或工具库（如spdlog、boost等）以外的任何库函数，包括项目内 da-sha 库。

其下每一个模块目录必须以 **cpt** 开头，便于导入该模块时区分。

### da-sha [SHAre]

[sha 开发指南](https://u2c6hvagv9.feishu.cn/docx/doxcnc130tg8xpjbCyUtWs1SaNe)

项目共享模块。可在**不同业务之间**之间共享的模块。如果只会在自己的业务内共享，**不能**放入这个模块。

其下每一个模块目录必须以 **sha** 开头，便于导入该模块时区分。

### da-ent [ENTry]

[ent 开发指南](https://u2c6hvagv9.feishu.cn/docx/doxcnN1Z5uL4ZXTuDTc2XWx5xnh)

项目入口模块。该目录下的每一个目录都代表一类业务，业务下的每一个目录为单独的 Entry Project，其命名首三个字母必须为业务的缩写，业务必须为可执行文件，即有main函数。

其下每一个模块目录必须以 **ent** 开头。

例如有一个业务是和晶格结构设计相关，那么他的模块名为 ent-lattice-design，取其缩写为 LAD (LAttice Design)，那么对于保留特征的晶格设计业务，命名应为 'lad-feature-preserved-lattice'

### da-cmd [CoMmanD]

[cmd 开发指南](https://u2c6hvagv9.feishu.cn/docx/doxcnO3VXWw18rYXuQrUqXTC3k2)

每一个 Entry Project 都必须在该子模块中添加对应的脚本。一方面可以当做测试使用，二来其他使用者可以根据输入输出快速知道该功能的用途。

### da-bld [BLenDer]

[bld 开发指南](https://u2c6hvagv9.feishu.cn/docx/doxcnrHi3CIizV8DzzZ6eSB4nAh)

整个项目向 blender 暴露的出口。

## C++编写规范  

### 代码风格

项目 C++ 代码遵循 Google 规范，编码过程中请安装并配置 clang-format，它能够帮助你快速将C++代码文件整理得符合规范。

整个项目已经开启了 CppLint，请务必保证系统中有 Python 环境用以执行 cpplint.py。这个功能能够实现在编译前检查所有的 *.h 和 *.cpp 文件。

### 命名规范

由于之前的项目改造还没有完成，所以参考之前代码的时候请忽略 da-sha/sha-implicit-modeling.

#### 变量

- 局部变量、const常量： 全部小写，用下划线分割。如 `int hello_world = 3;`

- constexpr 常量：以`k`为前缀，以驼峰命名。如 `constexpr size_t kNumberOfShells = 5;`

- 全局变量常量：遵循上述局部变量规范。

- 类成员变量：遵循上述规范的同时，最后加下划线后缀。如 `const size_t width_;`

#### 函数

函数需以大写开头的驼峰命名法，即帕斯卡命名，如 `int ComputeSomething();`。语义上应以陈述句为主，并且非类内函数应斟酌考虑是否使用缩写，以免使用者的困惑。

Lambda 表达式虽然是变量，但是仍需要用函数的命名规范处理。

函数返回类型名如果很长，请用尾置返回类型：`VeryLongReturnType function();` -> `auto function() -> VeryLongReturnType;`

#### 类型

所有类型名均帕斯卡命名，并且语义上以名词为主，表示一类实体。

除了 class 定义的类型，使用关键字 `typedef`，`using` 等定义的，无论位于局部还是全局，也一律按照上述要求。

#### 宏定义

> 不推荐使用宏定义，如有泛型需求，请优先使用模板

全部大写，并以下划线分隔单词。有函数语义的宏定义依然需要陈述句命名。

## Git 使用

在启动开发自己的项目前，务必从 main 分支中新建一个自己的分支，并以自己在git中的用户名命名。

**绝对不可以在本地修改main分支，并严禁在主分支上push**，你能对 main 分支产生改动的唯一方式为在保证你的分支和 main 分支对齐后，提交 merge request，
等验证通过后合并。

git 的基本操作请参照 [](https://www.liaoxuefeng.com/wiki/896043488029600)

### 分支保持最新的方法（供参考）

由于主分支可能会在你分离出分支后发生变化，所以需要保证自己的分支源于最新主分支

在每次开发项目前，请 `git switch main` 转换到主分支，执行 `git pull` 保证主分支最新。 

然后切换到你自己的分支 `git switch <your branch name>`，执行 `git rebase main`。

运气好的话（主分支的修改与你修改的代码不冲突），将直接完成 rebase

如果出现一系列冲突，这个时候请配合 `git status` 命令查看需要修改的代码。 处理完一次冲突后，执行 `git rebase --continue` 继续 rebase，直到所有冲突解决完成。

解决完成冲突后，请再次运行一遍 `git rebase main` 以确保解决完成。
