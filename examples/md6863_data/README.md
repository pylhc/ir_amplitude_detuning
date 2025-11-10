# Measurement Data

This is the analysed amplitude detuning measurement data from MD8683 from 2022-06-24 in the LHC.
Only the kick- and bbq- files have been copied to save some space.

```bash
# Copy all kick*.tfs and bbq_ampdet.tfs files from source to target, preserving folder structure.
COPYSOURCE="/path/to/source/"
COPYTARGET="/path/to/target/"

find $COPYSOURCE -type f \( -name 'kick_*.tfs' -o -name 'bbq_ampdet.tfs' \) -exec sh -c 'mkdir -p "$COPYTARGET/$(dirname "{}")" && cp "{}" "$COPYTARGET/{}"' \;
```
