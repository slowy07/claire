version = "0.0.1"
author = "slowy07"
description = "library tensor array"
license = "MIT License"

srcDir = "src"

requires "nim >= 0.15.1", "nimblas >= 0.1.3"

## Testing task
proc test(name: string) =
  if not dirExist "bin":
    mkDir "bin"
  if not dirExist "nimcache":
    mkDir "nimcache"
  --run
  --nimcache: "nimcache"
  switch("out", ("./bin/" & name))
  setCommand "c", "claire_testing" & name & ".nim"

task test, "Run all test - internal":
  test "all_test"
