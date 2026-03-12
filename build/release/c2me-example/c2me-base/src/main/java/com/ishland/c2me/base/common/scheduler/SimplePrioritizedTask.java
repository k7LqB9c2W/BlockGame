package com.ishland.c2me.base.common.scheduler;

import com.ishland.flowsched.executor.LockToken;
import com.ishland.flowsched.executor.Task;

import java.util.Objects;

public class SimplePrioritizedTask extends Task {

    private final Runnable task;
    private final LockToken[] lockTokens;

    public SimplePrioritizedTask(Runnable task, LockToken[] lockTokens) {
        this.task = Objects.requireNonNull(task, "task");
        this.lockTokens = Objects.requireNonNull(lockTokens, "lockTokens");
    }

    @Override
    public void run(Runnable releaseLocks) {
        try {
            this.task.run();
        } finally {
            releaseLocks.run();
        }
    }

    @Override
    public void propagateException(Throwable t) {
        t.printStackTrace();
    }

    @Override
    public LockToken[] lockTokens() {
        return this.lockTokens;
    }
}
