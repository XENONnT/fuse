1.2.0 / 2024-04-11
------------------
* Specify lxml version (#185)
* Avoid duplicate code in downchunking (#180)
* Exclude PMT AP photons from raw_area_truth calculation (#176)
* Optional random seed as user input (#179)
* pre-commit autoupdate (#186)
* Automatic Plugin Documentation (#187)
* Embellish some codes (#188)
* Separate above- and below-gate drift of electrons (#178)
* Add photoionization plugins (#52)

1.1.0 / 2024-03-20
------------------
* Cleanup and refactor configurations (#163)
* Convert _photon_timings to integer values (#171)
* Debug entry_stop (#166 and #170)
* Fix a bug the pre-commit introduced and update the .git-blame-ignore-revs (#165)
* pre-commit autoupdate (#164)
* Make s2 AFT scaling optional (#161)
* Move eventid cut into uproot data loading (#159)
* Add flag to disable PMT afterpulsing (#158)
* Add option to enable or disable the noise (#157)
* Prevent empty instructions from locking NEST random seed (#156)
* Change raw_records to save_when TARGET (#148)
* Proposal to use pre-commit for continuous integration (#155)
* Changes in map interpolation methods (#152)
* Addition of truth plugins (#99)
* Handle missing corrections_version and cleanup (#143)
* Prevent empty instructions to lock NEST random seed (#146)
* Increase size of string fields in geant4_interactions (#142)
* No rechunking for microphysics summary & e_field NaNs to zero (#140)
* Fix csv timing (#137)
* Change electric field dtype to float32 (#135)
* CSV input plugin update (#132)
* Save electric field as float16 instead of int64 (#133)
* Add missing badges and tests (#127)
* Bring the documentation to readthedocs (#124)
* Some code improvements for better performance (#125)
* Simplify pytest and coveralls to shorten time usage (#123)

1.0.0 / 2024-02-08
-------------------
* First stable release
