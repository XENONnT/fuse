1.5.0 / 2025-05-21
------------------
* Only simulate a part of the PI train by  (https://github.com/XENONnT/fuse/pull/287)
* typo fixes, code clarificaiton (https://github.com/XENONnT/fuse/pull/290)
* Patch cmake version in nestpy (https://github.com/XENONnT/fuse/pull/302)
* Cleanup unused configs (https://github.com/XENONnT/fuse/pull/301)
* Daq photon sorting (https://github.com/XENONnT/fuse/pull/297)
* Fuse compatibility with straxen (support both straxen v2 and v3)  (https://github.com/XENONnT/fuse/pull/298)
* Public config (https://github.com/XENONnT/fuse/pull/288)
* updating the fixed time separation (https://github.com/XENONnT/fuse/pull/289)
* Move context to fuse (https://github.com/XENONnT/fuse/pull/299)
* Clip FDC input positions (https://github.com/XENONnT/fuse/pull/304)

1.4.4 / 2025-01-25
------------------
* Be compatible with straxen >= 3 (https://github.com/XENONnT/fuse/pull/282)
* Remove CMT URLs, drop python 3.9 (https://github.com/XENONnT/fuse/pull/285)


1.4.3 / 2025-01-13
------------------
* Little bugfix for CSV generators (https://github.com/XENONnT/fuse/pull/273)
* Constraint strax(en) to be less than 2.0.0(3.0.0) (https://github.com/XENONnT/fuse/pull/276)
* Use master for docformatter (https://github.com/XENONnT/fuse/pull/277)
* Fix chunk ends late issue (https://github.com/XENONnT/fuse/pull/275)
* [pre-commit.ci] pre-commit autoupdate (https://github.com/XENONnT/fuse/pull/274)

1.4.2 / 2024-10-16
------------------
* utilix>0.9 compatibility (https://github.com/XENONnT/fuse/pull/267)
* [pre-commit.ci] pre-commit autoupdate (https://github.com/XENONnT/fuse/pull/266)
* Import MongoDownloader from utilix (https://github.com/XENONnT/fuse/pull/271)
* Remove WFSim connection (https://github.com/XENONnT/fuse/pull/268)
* Area correction override (https://github.com/XENONnT/fuse/pull/269)

1.4.1 / 2024-09-17
------------------
* pulse_id type casting inconsistency (#260)
* Make sure CDF goes to 1 smoothly (#261)
* Option to replace maps by constant dummy map (#262)

1.4.0 / 2024-09-10
------------------
* Efficient memory chunking in input plugin (#207)
* fix numpy version (#243)
* [pre-commit.ci] pre-commit autoupdate (#242)
* Add lineage clustering algorithm (#190)
* Keep interactions even if outside NEST validity (#241)
* Update BetaYields (#192)
* Add option to distribute Geant4 events with fixed timing (#225)
* [pre-commit.ci] pre-commit autoupdate (#244)
* Update mass and charge dtypes to int16 (#246)
* Fix plugin warning/debug/info logging (#248)
* [pre-commit.ci] pre-commit autoupdate (#250)
* Exciton ratio fix (#251)
* Poetry does not understand `requires-python` (#252)
* [pre-commit.ci] pre-commit autoupdate (#253)
* [pre-commit.ci] pre-commit autoupdate (#255)
* [pre-commit.ci] pre-commit autoupdate (#256)
* Update yields and clustering (#245)
* Take yield and width parameters from config file (#257)
* Faster lineage clustering (#258)

1.3.0 / 2024-06-10
------------------
* Use straxen documentation building functions to avoid duplicated codes (#218)
* Replace scintillation time clipping by truncation (#220)
* Fix `ChunkCsvInput` data_type of chunk (#223)
* Add descriptions of dtypes in `tagged_clusters` (#227)
* Modify the definitions in peak truth to be consistent with appletree (#226)
* Raw Records simulation acceleration in fuse (#228)
* Fix a small bug in peak truth (#232)
* Make `pulse_id` continuous in chunks (#233)
* Keep negative gain photons (#231)
* Remove unused codes and fix nestpy commit hash (#237)
* Option to override args for NR scintillation delay (#235)

1.2.3 / 2024-05-16
------------------
* Fix bug when building docs locally (#217)
* Increase test timeout to 30 minutes (#221)
* Add missing return to stop plugin in case of empty input (#216)
* Add more tests (#212)

1.2.2 / 2024-05-03
------------------
* Add Zenodo batch (f08d6ce)
* pre-commit autoupdate (#205)
* No photons from electrons outside (#200)
* Rename s1 photon hits output (#206)
* Change g4_fields dtype for t (#209)
* Add x_truth, y_truth, z_truth (#202)
* Add tight coincidence and peak tagging in `ClusterTagging` (#210)

1.2.1 / 2024-04-24
------------------
* pre-commit autoupdate (#191)
* Fix the `data_kind` of `RecordsTruth` (#194)
* Test full chain in multiple chunks and clean up input tests (#184)
* Add a dtypes manager (#195)
* Add pull_request_template.md (#197)
* Scale S2 pattern map to correct S2 AFT (#196)

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
