MODULE MainModule


	TASK PERS wobjdata wobjS:=[FALSE,TRUE,"",[[633.937,-143.322,-8.17262],[0.701809,0.0141639,0.00207583,0.712222]],[[0,0,0],[1,0,0,0]]];
	CONST robtarget Start:=[[609.08,0.34,274.86],[2.39407E-05,-0.655027,-0.755606,3.98471E-05],[-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pCircle:=[[31.52,28.58,6.82],[0.00302615,-0.621907,-0.783084,0.0015261],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pSquare:=[[34.25,155.68,6.97],[0.00302742,-0.621904,-0.783086,0.00152751],[0,0,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pStar:=[[90.98,98.16,8.66],[0.00303238,-0.621891,-0.783096,0.00155319],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pHex:=[[140.34,38.28,6.93],[0.00303278,-0.62189,-0.783098,0.00155401],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pTriangle:=[[147.89,148.96,8.51],[0.00303796,-0.621894,-0.783094,0.00156164],[0,-1,1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	CONST robtarget pickBase:=[[-0.52,1.12,1.83],[0.00763118,-0.994707,-0.101983,0.00995214],[-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
	TASK PERS tooldata toolss:=[TRUE,[[0.518716,91.6319,181.716],[1,0,0,0]],[0.5,[50,0,50],[1,0,0,0],0,0,0]];
	TASK PERS wobjdata wobjD:=[FALSE,TRUE,"",[[570.667,126.259,7.14651],[0.999962,-0.000713855,-0.00331164,-0.00798816]],[[0,0,0],[1,0,0,0]]];

    
      !--- Pick orientation (cup down, taught in wobjS; copied from pickBase)
    CONST orient   oPick  := [0.00854535,-0.680693,-0.732512,-0.00329158];
    CONST confdata cnfPick := [-1,-1,0,0];
    CONST extjoint ejPick := [9E+09,9E+09,9E+09,9E+09,9E+09,9E+09];  

    !=====================================================================
    !  ANGLE TEACH settings (used by TeachAll / TeachSquare ... below)
    !=====================================================================
    ! Drop point on the sheet, wobjS mm. OPERATOR: choose a free spot within
    ! +-200 mm that the camera sees, away from the marker and other parts,
    ! and away from where the arm stands at Start (Start is roughly above
    ! x +150, y +20 in wobjS, the arm comes in from +y).
    ! teach_angle.py checks the part is within 15 mm of this point: if you
    ! change nTeachX/nTeachY, pass the same values as --drop-x / --drop-y.
    LOCAL CONST num nTeachX := -120;
    LOCAL CONST num nTeachY := 60;
    LOCAL CONST num nTeachH := 10;       ! part height = pick z on the sheet [mm], as PICK_Z_MM
    LOCAL CONST num nTeachLift := 60;    ! approach/lift height, as PickAt/PlaceAt [mm]
    LOCAL CONST num nTeachClearZ := 200; ! z in wobjS the cup rises to after a drop [mm]
    LOCAL CONST num nTeachWait := 1.0;   ! vacuum build-up / release, as PickAt/PlaceAt [s]
    LOCAL CONST num nTeachTurn := 20;    ! test turn [deg]; turn + 3 must stay below 30 (hexagon)
    ! Extra depth to grip a SEATED part in its pocket [mm], positive = deeper
    ! (below the taught place target). 0 = grip at the place target itself.
    ! FIRST RUN: step the first TakeFromPocket in MANUAL REDUCED SPEED and
    ! check the cup really reaches the part top before raising this.
    LOCAL CONST num nPocketDepth := 0;

    
       !=====================================================================
    PROC main()
        VAR string sMsg;
        VAR num nNr;
        VAR num nX;
        VAR num nY;
        VAR num nZ;
        VAR num nAng;
        VAR bool bExit := FALSE;

        ! Reset outputs and move to the home point.
        Initialize;
        TPWrite "Waiting for PC (dashboard or live_pick --send)";
        ConnectToPC;
        TPWrite "Connected to PC.";

        WHILE NOT bExit DO
            ! Wait for one line from the PC and strip its ending.
            sMsg := TrimEnd(RobotClientReciveMessage());

            ! One line is one request; quit ends the loop.
            IF sMsg = "quit" THEN
                bExit := TRUE;
            ELSEIF NOT ParseMsg(sMsg, nNr, nX, nY, nZ, nAng) THEN
                TPWrite "Bad message: " + sMsg;
            ELSEIF nNr < 1 OR nNr > 5 THEN
                TPWrite "Slot must be 1..5, got " + NumToStr(nNr, 0);
            ! Reject any point outside the usable sheet area.
            ELSEIF nX < -200 OR nX > 200 OR nY < -200 OR nY > 200 OR nZ < 0 OR nZ > 40 THEN
                TPWrite "Outside the sheet - ignored.";
            ELSEIF nAng < -180 OR nAng > 180 THEN
                TPWrite "Angle must be -180..180 - ignored.";
            ELSE
                TPWrite "Pick X=" + NumToStr(nX, 1) + " Y=" + NumToStr(nY, 1) + " Z=" + NumToStr(nZ, 1) + " A=" + NumToStr(nAng, 1);
                TPWrite "Press the yellow button to pick.";
                ! Lamp asks the operator to confirm before the pick.
                Set doYellowLamp;
                WaitDI diYellowBtn, 0;
                WaitDI diYellowBtn, 1;
                Reset doYellowLamp;

                PickAt nX, nY, nZ, nAng;
                PlaceAt GetPlaceTarget(nNr);
                TPWrite "Done.";
            ENDIF
        ENDWHILE

        RobotClientCloseAndDisconnect;
        Reset doValve1;
        MoveJ Start, v300, fine, toolss\WObj:=wobj0;
        ! Link closed, vacuum off, arm at home.
        TPWrite "Program finished.";

    ERROR
        ! The PC closed the link (live_pick.py does this after sending, the
        ! dashboard when you quit it): reconnect and wait for the next line.
        IF ERRNO = ERR_SOCK_CLOSED THEN
            TPWrite "PC disconnected - waiting for it again...";
            RobotClientCloseAndDisconnect;
            ConnectToPC;
            TPWrite "Connected to PC.";
            RETRY;
        ENDIF
    ENDPROC

    !=====================================================================
    !  Try to connect every 2 s until the PC is listening.
    PROC ConnectToPC()
        RobotAsClientConnect;
    ERROR
        RobotClientCloseAndDisconnect;
        WaitTime 2;
        RETRY;
    ENDPROC

    !=====================================================================
    PROC Initialize()
        Reset doValve1;
        Reset doYellowLamp;
        ConfJ \Off;
        ConfL \Off;
        MoveJ Start, v300, fine, toolss\WObj:=wobj0;
    ENDPROC

    !=====================================================================
    !  The point nH mm above pT, along the z axis of the work object the
    !  Move instruction names (straight up off the sheet or the tray).
    FUNC robtarget Above(robtarget pT, num nH)
        RETURN [[pT.trans.x, pT.trans.y, pT.trans.z + nH], pT.rot, pT.robconf, pT.extax];
    ENDFUNC

    !=====================================================================
    !  Pick at (x, y, z) in wobjS, gripper turned nAng about its own axis.
    PROC PickAt(num nX, num nY, num nZ, num nAng)
        VAR robtarget pTarget;
        VAR num nPickDepth := 0;     ! extra depth below z (negative = deeper)
        VAR num nLift      := 60;    ! lift height [mm]
        VAR num nWait      := 1.0;   ! vacuum build-up [s]

        ! Position from the PC; cup-down orientation and config from pickBase,
        ! which was taught in wobjS (so the orientation is in the right frame).
        pTarget := pickBase;
        pTarget.trans := [nX, nY, nZ + nPickDepth];
        ! Turn the cup by the angle sent by Python.
        pTarget := RelTool(pTarget, 0, 0, 0 \Rz:=nAng);

        ! Approach from above, grip the part, then lift back up.
        MoveJ Above(pTarget, nLift), v300, z20, toolss\WObj:=wobjS;
        MoveL pTarget, v50, fine, toolss\WObj:=wobjS;
        Set doValve1;
        WaitTime nWait;
        MoveL Above(pTarget, nLift), v50, z20, toolss\WObj:=wobjS;
    ENDPROC

    !=====================================================================
    !  Place at the taught pocket orientation. No extra turn here: the pick
    !  already lined the part up with the cup, so turning again would
    !  rotate the part by the same angle a second time.
    PROC PlaceAt(robtarget pTarget)
        VAR num nLift := 60;
        VAR num nWait := 1.0;

        ! Same steps as the pick, but release into the pocket.
        MoveJ Above(pTarget, nLift), v300, z20, toolss\WObj:=wobjD;
        MoveL pTarget, v50, fine, toolss\WObj:=wobjD;
        Reset doValve1;
        WaitTime nWait;
        MoveL Above(pTarget, nLift), v100, z20, toolss\WObj:=wobjD;
    ENDPROC

    !=====================================================================
    ! Slot 1 to 5 picks the pocket target for that part.
    FUNC robtarget GetPlaceTarget(num nNr)
        TEST Round(nNr)
        CASE 1:
            RETURN pCircle;
        CASE 2:
            RETURN pSquare;
        CASE 3:
            RETURN pHex;
        CASE 4:
            RETURN pTriangle;
        DEFAULT:
            RETURN pStar;
        ENDTEST
    ENDFUNC

    !=====================================================================
    !  Teaching helper: PP to Routine -> CheckTargets
    PROC CheckTargets()
        ConfJ \Off;
        ConfL \Off;
        MoveJ Start, v100, fine, toolss\WObj:=wobj0;
        TPWrite "Above CIRCLE";
        MoveJ Above(pCircle, 60), v100, fine, toolss\WObj:=wobjD;
        Stop;
        TPWrite "Above SQUARE";
        MoveJ Above(pSquare, 60), v100, fine, toolss\WObj:=wobjD;
        Stop;
        TPWrite "Above HEX";
        MoveJ Above(pHex, 60), v100, fine, toolss\WObj:=wobjD;
        Stop;
        TPWrite "Above TRIANGLE";
        MoveJ Above(pTriangle, 60), v100, fine, toolss\WObj:=wobjD;
        Stop;
        TPWrite "Above STAR";
        MoveJ Above(pStar, 60), v100, fine, toolss\WObj:=wobjD;
        Stop;
        MoveJ Start, v100, fine, toolss\WObj:=wobj0;
    ENDPROC

    !=====================================================================
    !  ANGLE TEACH: the robot takes each SEATED part out of its pocket and
    !  sets it on the sheet at (nTeachX, nTeachY) with the normal pick pose,
    !  once with no turn and once turned nTeachTurn. teach_angle.py measures
    !  both with the camera and works out the turn direction and the zero
    !  angle for each shape. PP to Routine -> TeachAll.
    !  Re-run it whenever pickBase or wobjS is re-taught.
    !=====================================================================
    PROC TeachAll()
        TeachShape 2;
        TeachShape 3;
        TeachShape 4;
        TeachShape 5;
        TPWrite "Angle teach finished. Run teach_angle.py --show";
    ENDPROC

    !=====================================================================
    !  nNr = slot: 2 square, 3 hexagon, 4 triangle, 5 star. Start with the
    !  part SEATED in its pocket and nothing else at the drop point.
    PROC TeachShape(num nNr)
        VAR num nAnswer;

        IF nNr < 2 OR nNr > 5 THEN
            TPWrite "TeachShape: slot must be 2..5 (circle has no angle)";
            RETURN;
        ENDIF
        ! Check the teach numbers before anything moves.
        IF NOT TeachPointOk() THEN
            RETURN;
        ENDIF
        ! Wrong part / rerun guard: the routine assumes the part is in its
        ! pocket and the drop point is free. Ask; anything but Yes aborts.
        TPErase;
        TPWrite "Teach " + ShapeName(nNr) + ":";
        TPWrite "Is the part SEATED in its pocket and";
        TPWrite "the drop point on the sheet EMPTY?";
        TPReadFK nAnswer, "Confirm", "Yes", stEmpty, stEmpty, stEmpty, "No";
        IF nAnswer <> 1 THEN
            TPWrite "TeachShape aborted - nothing moved.";
            RETURN;
        ENDIF
        ConfJ \Off;
        ConfL \Off;
        Reset doValve1;
        MoveJ Start, v100, fine, toolss\WObj:=wobj0;

        ! --- turn 0
        TPWrite "Teach " + ShapeName(nNr) + ": pocket -> sheet, turn 0";
        TakeFromPocket nNr;
        DropOnSheet 0;
        TPWrite "Run: teach_angle.py --turn 0 --shape " + ShapeName(nNr);
        TPWrite "then press PLAY";
        Stop;
        PickFromSheet 0;
        PlaceAt GetPlaceTarget(nNr);

        ! --- turn nTeachTurn
        TPWrite "Teach " + ShapeName(nNr) + ": pocket -> sheet, turn " + NumToStr(nTeachTurn, 0);
        TakeFromPocket nNr;
        DropOnSheet nTeachTurn;
        TPWrite "Run: teach_angle.py --turn " + NumToStr(nTeachTurn, 0) + " --shape " + ShapeName(nNr);
        TPWrite "then press PLAY";
        Stop;
        PickFromSheet nTeachTurn;
        PlaceAt GetPlaceTarget(nNr);
        MoveJ Start, v100, fine, toolss\WObj:=wobj0;
        TPWrite ShapeName(nNr) + " back in its pocket.";
    ENDPROC

    !=====================================================================
    !  Same limits main() applies to a PC message, so the teach drops only
    !  where a normal pick could later go.
    LOCAL FUNC bool TeachPointOk()
        IF nTeachX < -200 OR nTeachX > 200 OR nTeachY < -200 OR nTeachY > 200 OR nTeachH < 0 OR nTeachH > 40 THEN
            TPWrite "AngleTeach: drop point outside the sheet - edit nTeachX/Y/H";
            RETURN FALSE;
        ENDIF
        IF nTeachTurn <= 0 OR nTeachTurn + 3 >= 30 THEN
            TPWrite "AngleTeach: nTeachTurn must be 1..26 deg";
            RETURN FALSE;
        ENDIF
        IF nPocketDepth < 0 OR nPocketDepth > 10 THEN
            TPWrite "AngleTeach: nPocketDepth must be 0..10 mm";
            RETURN FALSE;
        ENDIF
        IF nTeachClearZ < nTeachH + nTeachLift THEN
            TPWrite "AngleTeach: nTeachClearZ must be above nTeachH + nTeachLift";
            RETURN FALSE;
        ENDIF
        RETURN TRUE;
    ENDFUNC

    !=====================================================================
    !  The ONE sheet target for both the drop and the pick-back, built exactly
    !  as PickAt builds it (pickBase orientation, RelTool \Rz in tool z), so
    !  the angle the camera sees is the one a PickAt with nAng would undo.
    LOCAL FUNC robtarget TeachTarget(num nAng)
        VAR robtarget pT;

        pT := pickBase;
        pT.trans := [nTeachX, nTeachY, nTeachH];
        pT := RelTool(pT, 0, 0, 0 \Rz:=nAng);
        RETURN pT;
    ENDFUNC

    !=====================================================================
    !  Vacuum-pick the seated part at the taught place target (wobjD),
    !  nPocketDepth deeper. Same approach, speeds and wait as PlaceAt, with
    !  the valve set instead. FIRST RUN: step this in manual reduced speed.
    LOCAL PROC TakeFromPocket(num nNr)
        VAR robtarget pT;
        VAR robtarget pGrip;

        ! Get the pocket point before applying the grip depth.
        pT := GetPlaceTarget(nNr);
        pGrip := Offs(pT, 0, 0, -nPocketDepth);
        MoveJ Above(pT, nTeachLift), v300, z20, toolss\WObj:=wobjD;
        MoveL pGrip, v50, fine, toolss\WObj:=wobjD;
        Set doValve1;
        WaitTime nTeachWait;
        MoveL Above(pT, nTeachLift), v50, z20, toolss\WObj:=wobjD;
    ENDPROC

    !=====================================================================
    !  Release the part on the sheet at TeachTarget(nAng), then get out of
    !  the camera picture:
    !   1. straight up (MoveL, fine) to nTeachClearZ, so the cup cannot drag
    !      or nudge the released part sideways while it moves away;
    !   2. then MoveJ to Start. Straight up alone is NOT enough: the cup and
    !      arm would still be right above the part and hide it from the
    !      camera. Start is used because it is the known safe home; the drop
    !      point must be chosen so the arm at Start does not cover it
    !      (teach_angle.py refuses if it does not see exactly one part).
    LOCAL PROC DropOnSheet(num nAng)
        VAR robtarget pT;
        VAR robtarget pClear;

        pT := TeachTarget(nAng);
        pClear := pT;
        ! Keep x and y, raise only the height.
        pClear.trans.z := nTeachClearZ;
        MoveJ Above(pT, nTeachLift), v300, z20, toolss\WObj:=wobjS;
        MoveL pT, v50, fine, toolss\WObj:=wobjS;
        Reset doValve1;
        WaitTime nTeachWait;
        MoveL pClear, v50, fine, toolss\WObj:=wobjS;
        MoveJ Start, v300, fine, toolss\WObj:=wobj0;
    ENDPROC

    !=====================================================================
    !  Pick the part back from the SAME robtarget it was dropped at (same
    !  TeachTarget(nAng)), so the part returns to the cup exactly as it left
    !  and PlaceAt seats it in the pocket again. Same steps as PickAt.
    LOCAL PROC PickFromSheet(num nAng)
        VAR robtarget pT;

        pT := TeachTarget(nAng);
        MoveJ Above(pT, nTeachLift), v300, z20, toolss\WObj:=wobjS;
        MoveL pT, v50, fine, toolss\WObj:=wobjS;
        Set doValve1;
        WaitTime nTeachWait;
        MoveL Above(pT, nTeachLift), v50, z20, toolss\WObj:=wobjS;
    ENDPROC

    !=====================================================================
    LOCAL FUNC string ShapeName(num nNr)
        TEST Round(nNr)
        CASE 2:
            RETURN "square";
        CASE 3:
            RETURN "hexagon";
        CASE 4:
            RETURN "triangle";
        CASE 5:
            RETURN "star";
        DEFAULT:
            RETURN "circle";
        ENDTEST
    ENDFUNC

    !=====================================================================
    !  One shape at a time: PP to Routine -> TeachSquare / TeachHexagon /
    !  TeachTriangle / TeachStar (PP to Routine only lists routines without
    !  parameters, so TeachShape itself cannot be picked there).
    PROC TeachSquare()
        TeachShape 2;
    ENDPROC

    PROC TeachHexagon()
        TeachShape 3;
    ENDPROC

    PROC TeachTriangle()
        TeachShape 4;
    ENDPROC

    PROC TeachStar()
        TeachShape 5;
    ENDPROC

    !=====================================================================
    !  Strip trailing CR / LF from a received line
    FUNC string TrimEnd(string s)
        VAR string t;
        t := s;
        WHILE StrLen(t) > 0 AND (StrPart(t, StrLen(t), 1) = "\0A" OR StrPart(t, StrLen(t), 1) = "\0D") DO
            t := StrPart(t, 1, StrLen(t) - 1);
        ENDWHILE
        RETURN t;
    ENDFUNC

    !=====================================================================
    !  Split "slot;x;y;z;angle" into five numbers. FALSE if malformed.
    FUNC bool ParseMsg(string s, INOUT num nr, INOUT num x, INOUT num y, INOUT num z, INOUT num a)
        VAR num p1;
        VAR num p2;
        VAR num p3;
        VAR num p4;
        VAR bool ok1;
        VAR bool ok2;
        VAR bool ok3;
        VAR bool ok4;
        VAR bool ok5;

        p1 := StrFind(s, 1, ";");
        IF p1 > StrLen(s) RETURN FALSE;
        p2 := StrFind(s, p1 + 1, ";");
        IF p2 > StrLen(s) RETURN FALSE;
        p3 := StrFind(s, p2 + 1, ";");
        IF p3 > StrLen(s) RETURN FALSE;
        p4 := StrFind(s, p3 + 1, ";");
        IF p4 > StrLen(s) RETURN FALSE;

        ! Turn each text field into a number.
        ok1 := StrToVal(StrPart(s, 1, p1 - 1), nr);
        ok2 := StrToVal(StrPart(s, p1 + 1, p2 - p1 - 1), x);
        ok3 := StrToVal(StrPart(s, p2 + 1, p3 - p2 - 1), y);
        ok4 := StrToVal(StrPart(s, p3 + 1, p4 - p3 - 1), z);
        ok5 := StrToVal(StrPart(s, p4 + 1, StrLen(s) - p4), a);
        RETURN ok1 AND ok2 AND ok3 AND ok4 AND ok5;
    ENDFUNC

ENDMODULE
