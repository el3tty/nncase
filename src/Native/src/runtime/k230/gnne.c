#include <stdio.h>
#include "gnne.h"

static volatile gnne_reg_file_t *gnne_regs;

void gnne_set_base(volatile void *addr) {
    gnne_regs = (volatile gnne_reg_file_t *)((char *)addr + GNNE_ICACHE_CFG_OFFSET);
}

void gnne_ctrl_set(uint64_t value) {
    gnne_regs->ctrl.data[1] = value;
}

void gnne_clear_cpu_intr(void) {
    gnne_ctrl_set(GNNE_CTRL_CPU_INTR_CLEAR);
}

uint64_t gnne_disable(void) {
    uint64_t result = gnne_ctrl_set(GNNE_CTRL_ENABLE_CLEAR);

    while (gnne_regs->status.bits.reset_status)
        ;

    return result;
}

uint64_t gnne_init()
{
  return gnne_disable();
}

int gnne_enable(int32_t pc_start, int32_t pc_end, int32_t pc_breakpoint) {

    if (gnne_regs->status.bits.kpu_work_status) {
        return 1;
    }

    gnne_regs->pc_cfg.bits.start_pc_addr_reg = pc_start;
    gnne_regs->pc_cfg.bits.end_pc_addr_reg = pc_end;
    gnne_regs->pc_cfg.bits.breakpoint_pc_addr_reg = pc_breakpoint;

    gnne_ctrl_set(GNNE_CTRL_ENABLE_SET | GNNE_CTRL_DEBUG_MODE_SET);
    return 0;
}

int gnne_resume(gnne_ctrl_function_t resume_mode, int32_t pc_start) {
    if (gnne_regs->status.bits.kpu_work_status != 2) {
        return 1;
    }

    if ((resume_mode & ~0x80ULL) == GNNE_CTRL_CPU_RESUME_MODE_1) {
        // handles MODE_1 and MODE_3 alike
        gnne_regs->pc_cfg.bits.start_pc_addr_reg = pc_start;
        gnne_ctrl_set(resume_mode);
        return 0;
    } else if (resume_mode == GNNE_CTRL_CPU_RESUME_MODE_0) {
        gnne_ctrl_set(GNNE_CTRL_CPU_RESUME_MODE_0);
        return 0;
    } else {
        printf("the resume mode(0x%lx) is not supported\n", resume_mode);
        return 2;
    }
}

gnne_status gnne_get_status(void) {
    return gnne_regs->status;
}

void gnne_dump_status(void) {
    gnne_status s = gnne_regs->status;

    printf("load_que_status = %u\n", s.bits.load_que_satus);
    printf("store_que_status = %u\n", s.bits.store_que_status);
    printf("dm_que_status = %u\n", s.bits.dm_que_status);
    printf("pu_que_status = %u\n", s.bits.pu_que_status);
    printf("mfu_que_status = %u\n", s.bits.mfu_que_status);
    printf("load_module_status = %u\n", s.bits.load_module_status);
    printf("store_module_status = %u\n", s.bits.store_module_status);
    printf("dm_module_status = %u\n", s.bits.dm_module_status);
    printf("pu_module_status = %u\n", s.bits.pu_module_status);
    printf("mfu_module_status = %u\n", s.bits.mfu_module_status);
    printf("version = %u\n", s.bits.version);
    printf("kpu_work_status = %u\n", s.bits.kpu_work_status);
    printf("exception_status = %u\n", s.bits.exception_status);
    printf("reset_status = %u\n", s.bits.reset_status);
    printf("intr_status = %u\n", s.bits.intr_status);
    printf("intr_num = %u\n", s.bits.intr_num);
}

void gnne_dump_pc(void) {
    printf("dec_pc = 0x%08x\n", gnne_regs->dec_ld_st_mfu_pc.bits.dec_pc);
    printf("load_pc = 0x%08x\n", gnne_regs->dec_ld_st_mfu_pc.bits.load_pc);
    printf("store_pc = 0x%08x\n", gnne_regs->dec_ld_st_mfu_pc.bits.store_pc);
    printf("mfu_pc = 0x%08x\n", gnne_regs->dec_ld_st_mfu_pc.bits.mfu_pc);
    printf("pu_pc = 0x%08x\n", gnne_regs->pu_pc.bits.pu_pc);
    printf("dw_pc = 0x%08x\n", gnne_regs->pu_pc.bits.dw_pc);
    printf("act0_pc = 0x%08x\n", gnne_regs->pu_pc.bits.act0_pc);
    printf("act1_pc = 0x%08x\n", gnne_regs->pu_pc.bits.act1_pc);
    printf("dm_w_pc = 0x%08x\n", gnne_regs->dm_pc.bits.dm_w_pc);
    printf("dm_if_pc = 0x%08x\n", gnne_regs->dm_pc.bits.dm_if_pc);
    printf("dm_psum_pc = 0x%08x\n", gnne_regs->dm_pc.bits.dm_psum_pc);
    printf("dm_act_pc = 0x%08x\n", gnne_regs->dm_pc.bits.dm_act_pc);
}

void gnne_dump_ccr(void) {
    gnne_ccr_status ccr = gnne_regs->ccr_status;

    printf("ccr0 = %u\n",  ccr.bits.ccr0);
    printf("ccr1 = %u\n",  ccr.bits.ccr1);
    printf("ccr2 = %u\n",  ccr.bits.ccr2);
    printf("ccr3 = %u\n",  ccr.bits.ccr3);
    printf("ccr4 = %u\n",  ccr.bits.ccr4);
    printf("ccr5 = %u\n",  ccr.bits.ccr5);
    printf("ccr6 = %u\n",  ccr.bits.ccr6);
    printf("ccr7 = %u\n",  ccr.bits.ccr7);
    printf("ccr8 = %u\n",  ccr.bits.ccr8);
    printf("ccr9 = %u\n",  ccr.bits.ccr9);
    printf("ccr10 = %u\n", ccr.bits.ccr10);
    printf("ccr11 = %u\n", ccr.bits.ccr11);
    printf("ccr12 = %u\n", ccr.bits.ccr12);
    printf("ccr13 = %u\n", ccr.bits.ccr13);
    printf("ccr14 = %u\n", ccr.bits.ccr14);
    printf("ccr15 = %u\n", ccr.bits.ccr15);
    printf("ccr16 = %u\n", ccr.bits.ccr16);
    printf("ccr17 = %u\n", ccr.bits.ccr17);
    printf("ccr18 = %u\n", ccr.bits.ccr18);
    printf("ccr19 = %u\n", ccr.bits.ccr19);
    printf("ccr20 = %u\n", ccr.bits.ccr20);
    printf("ccr21 = %u\n", ccr.bits.ccr21);
    printf("ccr22 = %u\n", ccr.bits.ccr22);
    printf("ccr23 = %u\n", ccr.bits.ccr23);
    printf("ccr24 = %u\n", ccr.bits.ccr24);
    printf("ccr25 = %u\n", ccr.bits.ccr25);
    printf("ccr26 = %u\n", ccr.bits.ccr26);
    printf("ccr27 = %u\n", ccr.bits.ccr27);
    printf("ccr28 = %u\n", ccr.bits.ccr28);
    printf("ccr29 = %u\n", ccr.bits.ccr29);
    printf("ccr30 = %u\n", ccr.bits.ccr30);
    printf("ccr31 = %u\n", ccr.bits.ccr31);
}

uint64_t gnne_get_pc(void) {
    return gnne_regs->dec_ld_st_mfu_pc.bits.dec_pc;
}

gnne_ctrl gnne_get_ctrl(void) {
    return gnne_regs->ctrl;
}

gnne_dec_ld_st_mfu_pc gnne_get_dec_ld_st_mfu_pc(void) {
    return gnne_regs->dec_ld_st_mfu_pc;
}

gnne_pu_pc gnne_get_pu_pc(void) {
    return gnne_regs->pu_pc;
}

gnne_dm_pc gnne_get_dm_pc(void) {
    return gnne_regs->dm_pc;
}

gnne_ccr_status gnne_get_ccr_status(void) {
    return gnne_regs->ccr_status;
}

uint32_t gnne_get_time_out(void) {
    return gnne_regs->time_out.bits.time_out_value;
}

void gnne_set_time_out(uint32_t val) {
    gnne_regs->time_out.bits.time_out_value = val;
}

uint32_t gnne_get_ai2d_pc(void) {
    return gnne_regs->ai2d_pc.bits.ai2d_pc_addr;
}

void gnne_set_ai2d_pc(uint32_t val) {
    gnne_regs->ai2d_pc.bits.ai2d_pc_addr = val;    
}