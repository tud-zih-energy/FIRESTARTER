#include <firestarter/Environment/X86/X86Environment.hpp>

using namespace firestarter::environment::x86;

// this is only implemented for 64 bit.
unsigned long long X86Environment::timestamp(void) {
  unsigned long long reg_a, reg_d;

  if (!this->hasRdtsc()) {
    return 0;
  }

  __asm__ __volatile__("rdtsc;" : "=a"(reg_a), "=d"(reg_d));
  return (reg_d << 32) | (reg_a & 0xffffffffULL);
}

void X86Environment::cpuid(unsigned long long *a, unsigned long long *b,
                           unsigned long long *c, unsigned long long *d) {
  unsigned long long reg_a, reg_b, reg_c, reg_d;

  __asm__ __volatile__("cpuid;"
                       : "=a"(reg_a), "=b"(reg_b), "=c"(reg_c), "=d"(reg_d)
                       : "a"(*a), "b"(*b), "c"(*c), "d"(*d));
  *a = reg_a;
  *b = reg_b;
  *c = reg_c;
  *d = reg_d;
}

bool X86Environment::hasRdtsc() {
  unsigned long long a = 0, b = 0, c = 0, d = 0;

  a = 0;
  this->cpuid(&a, &b, &c, &d);
  if (a >= 1) {
    a = 1;
    this->cpuid(&a, &b, &c, &d);
    if ((int)d & (1 << 4)) {
      return true;
    }
  }

  return false;
}

bool X86Environment::hasInvariantRdtsc(void) {
  unsigned long long a = 0, b = 0, c = 0, d = 0;

  if (this->hasRdtsc()) {

    /* TSCs are usable if CPU supports only one frequency in C0 (no
       speedstep/Cool'n'Quite)
       or if multiple frequencies are available and the constant/invariant TSC
       feature flag is set */

    if (0 == this->vendor.compare("GenuineIntel")) {
      /*check if Powermanagement and invariant TSC are supported*/
      a = 1;
      this->cpuid(&a, &b, &c, &d);
      /* no Frequency control */
      if ((!(d & (1 << 22))) && (!(c & (1 << 7)))) {
        return true;
      }
      a = 0x80000000;
      this->cpuid(&a, &b, &c, &d);
      if (a >= 0x80000007) {
        a = 0x80000007;
        this->cpuid(&a, &b, &c, &d);
        /* invariant TSC */
        if (d & (1 << 8)) {
          return true;
        }
      }
    }

    if (0 == this->vendor.compare("AuthenticAMD")) {
      /*check if Powermanagement and invariant TSC are supported*/
      a = 0x80000000;
      this->cpuid(&a, &b, &c, &d);
      if (a >= 0x80000007) {
        a = 0x80000007;
        this->cpuid(&a, &b, &c, &d);

        /* no Frequency control */
        if ((!(d & (1 << 7))) && (!(d & (1 << 1)))) {
          return true;
        }
        /* invariant TSC */
        if (d & (1 << 8)) {
          return true;
        }
      }
      /* assuming no frequency control if cpuid does not provide the extended
         function to test for it */
      else {
        return true;
      }
    }
  }

  return false;
}
