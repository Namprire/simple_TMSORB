from setuptools import find_packages, setup

package_name = 'mono_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/run_mono_slam.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotech',
    maintainer_email='doi-kanichi823@g.ecc.u-tokyo.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dataset_publisher = mono_slam.dataset_publisher:main',
            'slam_node = mono_slam.slam_node:main',
            'trajectory_logger_node = mono_slam.trajectory_logger_node:main',
        ],
    },
)
