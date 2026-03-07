from setuptools import setup
import os
from glob import glob

package_name = 'package_integrity_inference'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools', 'ultralytics', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='Caaren Amirian',
    maintainer_email='camirian@outlook.com',
    description='ROS 2 edge inference node for YOLOv8 classification of package integrity',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_inference_node = package_integrity_inference.yolo_inference_node:main'
        ],
    },
)
