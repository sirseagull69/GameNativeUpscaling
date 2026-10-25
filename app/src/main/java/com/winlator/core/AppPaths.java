package com.winlator.core;

import android.content.Context;

/**
 * Upstream GameNative hardcodes "/data/data/app.gamenative/..." in a few places
 * (container defaults, drive E:, mediaconv dump paths). For builds with a different
 * applicationId (e.g. the "ki" test build, app.gamenative.ki) those paths point into
 * another app's private storage. This rewrites them to this app's real data dir.
 * For the normal app.gamenative package it is a no-op.
 */
public final class AppPaths {
    private static final String[] UPSTREAM_PREFIXES = {
        "/data/data/app.gamenative/",
        "/data/user/0/app.gamenative/",
    };
    private static volatile String dataDirPrefix = null;

    private AppPaths() {}

    public static void init(Context context) {
        String pkg = context.getPackageName();
        if ("app.gamenative".equals(pkg)) {
            dataDirPrefix = null;
            return;
        }
        dataDirPrefix = context.getApplicationInfo().dataDir + "/";
    }

    /** Rewrites upstream-package absolute paths inside {@code value}; null-safe. */
    public static String fix(String value) {
        String prefix = dataDirPrefix;
        if (prefix == null || value == null) return value;
        String out = value;
        for (String upstream : UPSTREAM_PREFIXES) {
            if (out.contains(upstream)) out = out.replace(upstream, prefix);
        }
        return out;
    }

    public static boolean isRenamedBuild() {
        return dataDirPrefix != null;
    }
}
