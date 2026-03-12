package com.ishland.c2me.base.common.scheduler;

import com.ishland.flowsched.executor.Task;
import it.unimi.dsi.fastutil.objects.ReferenceArrayList;

import java.util.Objects;

public abstract class AbstractPosAwarePrioritizedTask extends Task {

    protected final ReferenceArrayList<Runnable> postExec = new ReferenceArrayList<>(4);
    private final long pos;

    public AbstractPosAwarePrioritizedTask(long pos) {
        this.pos = pos;
    }

    public long getPos() {
        return this.pos;
    }

    public void addPostExec(Runnable runnable) {
        synchronized (this.postExec) {
            postExec.add(Objects.requireNonNull(runnable));
        }
    }
}
