#!/usr/bin/env python

# This script runs all boost unit tests in a particular directory.
# For each test file outputs a report file with added .xml extension.
# The test executable should end with "Tests" or "Tests.exe" suffix

import os
import argparse
import subprocess

def runBoostUnitTests(testDir, outputDir):
    for test in os.listdir(testDir):    
        # skipping nested directories
        if not os.path.isfile(os.path.join(testDir, test)):
            continue
        
        # running the test with correct suffix
        if test.lower().endswith("tests.exe") or test.lower().endswith("tests"):
            outputFile = os.path.join(outputDir, test + ".xml")
            print "Running test executable %s" % test + " with result in %s" % outputFile
            subprocess.check_call([os.path.join(testDir, test), "--log_format=XML", "--log_sink=%s" % outputFile, "--log_level=all", "--report_level=no"])

def main():
    parser = argparse.ArgumentParser(description="Runs all boost unit tests in the directory")
    parser.add_argument('--testdir', help='Test directory where all tests reside', required=True)
    parser.add_argument('--outputdir', help='Output directory for test results', required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    runBoostUnitTests(args.testdir, args.outputdir)

main()