import mrcnn.model as modellib
from mrcnn.config import Config
from multiprocessing import Process, Queue, Event, log_to_stderr, get_logger
import logging

class NeuralNetService:
    """
    This class starts a NeuralNet as a serverlike daemon and enables multiprocessing. It gives you the option to
    wait for the Prediction of your NeuralNet when you need them.

    :param netconf: NeuralNetConfiguration, adapted to your weights
    :param ptweights: the absolute Path to your weights
    :param start: If set to True, will start the service on Initialisation. Default: False
    ##### use this methods #######
        predict: let the loaded model detecting the given images, waits till detection is ready
        add_prediction_task: adds an imgList to a task queue, which the NeuralNet will work on
        get_prediction_task: returns the prediction of a task, first imgList in first prediction out (FIFO)
        start_service: starts the process hosting the NeuralNet
        shut_down_service: shuts down the service, dose not check for unfinished tasks in task queue
        ....
    """
    def __init__(self, netconf, ptweights, start=False):

        # important for logging in the console #
        log_to_stderr()
        logger = get_logger()
        logger.setLevel(logging.INFO)

        # initalizing the service
        self.init_service(netconf, ptweights, start)

    def __del__(self):
        self.shut_down_service()

    def add_prediction_task(self, imgList):
        """
        Puts imgList as a new task in task queue. (FIFO)
        :param imgList: Liste der Bilder
        :return: None
        """
        self._input_queue.put(imgList)

    def get_prediction_task(self, block=True, timeout=None):
        """
        Gets the first prediction in the prediction queue(FIFO). A way for synchronization if the default parameters
        are used. Otherwise if the queue is empty and the time to wait is out raises "queue.Empty" exception.
        :param block: If set to False, it dose not wait.
        :param timeout: If block is set to True(default), it waits timeout seconds for something to return.
        :return: Predictions as a List
        """
        out = self._output_queue.get(block=block, timeout=timeout)
        return out


    def predict(self, imgList):
        """
        Same method as like in NeuralNet
        Puts the imgList in the queue and waits for it to be done.
        :param imgList: a list with the images to detect
        :return: Predictions of the imgList
        """
        self.add_prediction_task(imgList)
        return self.get_prediction_task()

    def _start_neuralnet_service(self, netConf, pathObjWeights, input_queue, output_queue, shut_down_event):
        """
        Method being run in the spawned process.
        :param netConf: NeuralNetConfiguration, adapted to your weights
        :param pathObjWeights: the absolute Path to your weights
        :param input_queue: Queue with Elements to do the Prediction
        :param output_queue: Queue with the Predictions
        :param shut_down_event: Event to shutdown the Process, not working yet
        :return: None
        """
        # configurations for tensorflow so the model only occupys as much space as needed
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)

        net = NeuralNet(netConf, pathObjWeights)
        while not shut_down_event.is_set():
            try:
                input = input_queue.get()
            except input_queue.Empty:
                continue
            if input == "END":
                break
            output_queue.put(net.predict(input))

    def init_service(self, netConf, ptweights, start=False):
        """
        Initiates a new process, the old one will be shutten down and overwritten.
        :param netConf: NeuralNetConfiguration, adapted to your weights
        :param ptweights: the absolute Path to your weights
        :param start: If set to True, will start the service on Initialisation. Default: False
        :return: None
        """

        self._shut_down_event = Event()     # if set, the while-loop closes
        self._input_queue = Queue()     # task queue, the NeuralNet is working on
        self._output_queue = Queue()    # prediction queue, the NeuralNet propagates the detections to

        if hasattr(self, '_process'):
            if self._process.is_alive():
                self.shut_down_service()

        self._process = Process(target=self._start_neuralnet_service,
                                args=(netConf, ptweights, self._input_queue, self._output_queue,
                                      self._shut_down_event),
                                daemon=True)
        if start:
            self.start_service()

    def start_service(self):
        self._process.start()

    def set_shut_down_event(self):
        """
        Starts to shut down the Service without waiting for it to be done.
        :return: None
        """
        self._shut_down_event.set()
        self.add_prediction_task("END")


    def shut_down_service(self):
        """
        Starts to shut down the Service and waits for it to be done.
        :return: None
        """
        self.set_shut_down_event()
        if self._process.is_alive():
            self._process.join()

class NeuralNet:
    """
    Just a class using mcrnn.model for more confortable way to work with your neural network
    netconf: NeuralNetConfiguration, adapted to your weights
    ptweights: the absolute Path to your weights
    ##### use this methods #######
        predict: let the loaded model detecting the given images
        ....
    """

    def __init__(self, netconf, ptweights):

        self.netConfig = netconf
        self._net = modellib.MaskRCNN(mode="inference", config=self.netConfig, model_dir=str(ptweights))
        self._net.load_weights(str(ptweights), by_name=True)

    def predict(self, imgList, verbose=None):
        """
        Let the loaded model detect the given images. Workflow: First in first out
        :param imgList: List of images to detect
        :param verbose: None
        :return: Returns a List with the results(dictionary format) of the detection
        """

        results = []    # list to fill with the results of the detection
        """ declaration results:
            - list of the results of the detection as dictionary, FIFO
            - results[i].keys(): ['rois', 'class_ids', 'scores', 'masks']
            - results[i]["rois"]: [[ 47 145 256 933]], list of coordinates of the bbox: [y1 x1 y2 x2]
            - results[i]["class_ids"]: [1], list of the detected ids
            - results[i]["scores"]: [1], list of the corresponding scores
            - results[i]["masks"]: ndarry with the masks in True/False format at the corresponding pixels
                -> shape = (350, 1066, 1) so (yi, xi, 1)
                -> HOW-TO convert in gray-scale image: imgBW = results[0]["masks"].astype(numpy.float)
                -> HOW-TO getting only first mask of imgList[i]: results[i]["masks"][:, :, 0]
        """
        # iterates over imgList and dose detection
        for x in imgList:
            results.append(self._net.detect([x], verbose=verbose)[0])
        return results

    def get_model(self):
        return self._net

class NeuralNetConfiguration(Config):
    """
    Configuration-Class for the NeuralNet. Please make your own class with similar structure and Values
    as like for training. All relevant Values you see here are used in an other Project.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "NeuralNet"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Anchore-Scaling for the first detection of ROIS
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # personal edits
    DETECTION_MIN_CONFIDENCE = 0.5

    def __init__(self, netConf):
        # personal init using a config-file
        self.BACKBONE = netConf["backBone"]
        self.CLASSES = netConf["classNames"]
        self.NUM_CLASSES = len(self.CLASSES)
        super().__init__()