package com.ishland.c2me.opts.dfc.common.gen;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class GenDumper {

    public static final File exportDir = new File("./cache/c2me-dfc");

    static {
        try {
            org.spongepowered.asm.util.Files.deleteRecursively(exportDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void dumpClass(String className, byte[] bytes) {
        File outputFile = new File(exportDir, "classes/" + className + ".class");
        outputFile.getParentFile().mkdirs();
        try {
            com.google.common.io.Files.write(bytes, outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Path dumpCL(String name, byte[] bytes) {
        File outputFile = new File(exportDir, "cl/" + name + ".cl");
        outputFile.getParentFile().mkdirs();
        try {
            com.google.common.io.Files.write(bytes, outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return outputFile.getAbsoluteFile().toPath();
    }

}
