
from image_classifier.data.image_classification import ImageClassificationDataset, ImageType

class MNISTDataset(ImageClassificationDataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        load_data: bool = True,
    ):
        super(MNISTDataset, self).__init__(
            img_dir=img_dir,
            annotations_file=annotations_file,
            img_size=(1, 28, 28),
            img_type=ImageType.GRAYSCALE,
            load_data=load_data,
        )