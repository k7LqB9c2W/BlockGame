package com.ishland.c2me.rewrites.chunksystem.common.structs;

import com.ishland.c2me.base.common.GlobalExecutors;
import io.reactivex.rxjava3.core.Scheduler;
import io.reactivex.rxjava3.schedulers.Schedulers;

import java.util.ArrayDeque;
import java.util.Queue;
import java.util.concurrent.Executor;

public class ChunkSystemExecutors {

    public static final ScopedValue<Queue<Runnable>> CONSOLIDATING_QUEUE = ScopedValue.newInstance();

    public static final Executor backingBackgroundExecutor = GlobalExecutors.prioritizedScheduler.executor(15);
    public static final Scheduler backgroundScheduler = Schedulers.from(backingBackgroundExecutor);
    public static final Executor consolidatingBackgroundExecutor = command -> {
        if (!CONSOLIDATING_QUEUE.isBound()) { // first entry
            consolidatingRoot(command);
        } else {
            Queue<Runnable> runnables = CONSOLIDATING_QUEUE.get();
            runnables.add(command);
        }
    };
    public static final Scheduler consolidatingBackgroundScheduler = Schedulers.from(consolidatingBackgroundExecutor);

    private static void consolidatingRoot(Runnable initialCommand) {
        ArrayDeque<Runnable> runnables = new ArrayDeque<>();
        runnables.add(initialCommand);
        consolidatingRoot0(runnables);
    }

    public static void consolidatingRoot0(ArrayDeque<Runnable> runnables) {
        backingBackgroundExecutor.execute(() -> {
            ScopedValue
                    .where(CONSOLIDATING_QUEUE, runnables)
                    .run(() -> {
                        while (!runnables.isEmpty()) {
                            try {
                                runnables.remove().run();
                            } catch (Throwable t) {
                                t.printStackTrace();
                            }
                        }
                    });
        });
    }

}
