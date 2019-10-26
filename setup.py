from setuptools import setup

setup(name='intersections-comp',
      description='Repo for Kaggle Comp: "BigQuery-Geotab Intersection Congestion"',
      author='Zach Griffith',
      author_email='griffitzd@gmail.com',
      packages=['intersections_comp'],
      setup_requires=['setuptools_scm'],
      install_requires=[
          'jupyter',
          'pandas',
          'scikit-learn'
      ],
      zip_safe=False,
      use_scm_version=True,
      include_package_data=True)

