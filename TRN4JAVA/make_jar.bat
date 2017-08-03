javac TRN4JAVA\Api.java 
jar cvmf MANIFEST.MF TRN4JAVA\TRN4JAVA.jar TRN4JAVA\*.class TRN4JAVA\Api.java -C native/ .
del TRN4JAVA\*.class 
copy TRN4JAVA\TRN4JAVA.jar "C:\Users\cazin\Dropbox\share_pablo\simulator\TRN4JAVA"
copy TRN4JAVA\TRN4JAVA.jar "C:\Users\cazin\git\scs\deps"