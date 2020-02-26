# Tensorflow wrapper for image segmentation
### Требования
* Tensorflow
* Protobuf
* Opencv 
* Abseil _(для тестов)_
#### Инструкция

* Простой способ установить  tensorflow - использовать [этот](https://github.com/leggedrobotics/tensorflow-cpp) репозиторий. 
Он использует предкомпилированные бинарники, так что савится тоже очень быстро.

* Для установки Protobuf я предлагаю использовать [это](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).
_Версия, которую я использую - 3.7.0_

* Стандартный способ установки Abseil [здесь](https://github.com/abseil/abseil-cpp)
 
 Для сборки используется Cmake.
 
 Cтандартный способ сборки проекта
 
 `mkdir build && cd build`
 
 `cmake .. && make -j`
 
 Соберётся статическая библиотека `build/src/libTF_WRAPPER_SEGMENTATION`, а так же бинарник - пример, 
 который эту либу исполользует `build/application/TF_SEGMENTATOR`
 
#### Настройка

Для настройки библиотеки есть конфигурационный файл `config.json`

В качестве альтернативы можно использовать метод `configure_wrapper` у класса `SegmentationWrapperBase`

* Параметр `"input size"` отвечает за размер, к которому будет приводится входное изображение. 
Важно помнить что этот размер напрямую зависит от того, какую нейросеть использовать.
* Параметр `"colors_path"` представляет из себя путь к .csv файлу, в котором хранятся данные для 
посторения цветовой маски изображения.
* Параметр `"pb_path"` представляет из себя путь к .pb(protobuf) файлу, в котором храниться структура и веса 
натренированной нейросети.
* Параметр `"input_node"` отвечает за входной узел натренированной нейросети.
* Параметр `"output_node"` отвечает за выходной узел натренированной нейросети соответственно.
### API
Для взаимодействия с библиотекой достаточно подключить её через Cmake и импортировать `"wrapper_base.h"`

1. Создать объект класса `SegmentationWrapperBase`
2. Вызвать метод `set_images({img_path})` с аргументом, которым является вектором путей к изображениям.
3. Вызвать метод `process_images()`
4. Вызвать метод `get_colored()`, если нужна цветная маска
5. Вызвать метод `get_indices()`, если нужна маска индексов

_Методы `get_colored` и `get_indices`_ вернут `std::vector<cv::Mat>`, который будет вектором с выходным изображением 
_(вектор для батча изображений в дальнейшем)_.

#### Пример
Файл `TF_SEGMENTATOR` является примером исользования библиотеки. Запускается с двумя аргументами:

* `-img` - путь к изображению
* `-colored` - true/false в зависимости от того, какой результат нужен.

Результат сохраняется в папке с бинаринком в файл `out.png`
