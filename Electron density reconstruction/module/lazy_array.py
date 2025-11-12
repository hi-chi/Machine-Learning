import numpy as np


class LazyArray:
    def __init__(self, initial_size, dtype=np.float32):
        self.initial_size = initial_size
        self.array = None
        self.shape_initialized = False

        self.dtype = dtype

    def _initialize_shape(self, tensor):
        tensor_shape = np.shape(tensor)
        self.array = np.zeros((self.initial_size, *tensor_shape), dtype=self.dtype)
        self.shape_initialized = True

    def __getitem__(self, index):
        if not self.shape_initialized:
            raise ValueError(
                "Массив еще не инициализирован. Установите размерности через оператор присваивания."
            )
        return self.array[index]

    def __setitem__(self, index, value):
        if not self.shape_initialized:
            self._initialize_shape(value)
            np.mean(self.array)

        self.array[index] = value

    def __repr__(self):
        if self.array is None:
            return f"<LazyArray (uninitialized), initial_size={self.initial_size}>"
        return repr(self.array)


if __name__ == '__main__':
    lazy_array = LazyArray(3)
    lazy_array[0] = np.array([[1, 2, 3], [4, 5, 6]])
    print(lazy_array[0])
