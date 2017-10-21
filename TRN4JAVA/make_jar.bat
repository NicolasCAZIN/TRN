javac TRN4JAVA\*.java
jar cvmf MANIFEST.MF TRN4JAVA.jar TRN4JAVA\*.class TRN4JAVA\*.java
del TRN4JAVA\*.class 
copy TRN4JAVA.jar "C:\Users\cazin\Dropbox\share_pablo\simulator\TRN4JAVA"
copy TRN4JAVA.jar "C:\Users\cazin\git\scs\deps"