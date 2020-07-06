#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
void timer_handler (int signum)
{
    static int count = 0;
    printf ("timer expired %d times\n", ++count);
}


int main ()
{
    struct sigaction sa;
    struct itimerval timer;

    /* Install timer_handler as the signal handler for SIGVTALRM. */
    memset (&sa, 0, sizeof (sa));
    sa.sa_handler = &timer_handler;
    sigaction (SIGVTALRM, &sa, NULL);

    /* Configure the timer to expire after 100000 msec... */
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 1000000;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 10000000;

    setitimer (ITIMER_VIRTUAL, &timer, NULL);

    /* Do busy work. */
    while (1);
}
