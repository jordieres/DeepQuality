-- Adminer 4.8.1 PostgreSQL 15.1 dump

\connect "deepquality";

DROP TABLE IF EXISTS "GR_QPValues";
CREATE TABLE "public"."GR_QPValues" (
    "COILID" character varying(254) NOT NULL,
    "TILEID" double precision NOT NULL,
    "MID" double precision NOT NULL,
    "PLANT" double precision NOT NULL,
    "SIDE" character varying(255) NOT NULL,
    "MEAN" double precision NOT NULL,
    "MAX" double precision NOT NULL,
    "MIN" double precision NOT NULL,
    "COUNT" double precision NOT NULL,
    "tstamp" timestamp DEFAULT now() NOT NULL
) WITH (oids = false);


DELIMITER ;;

CREATE TRIGGER "t_name" BEFORE UPDATE ON "public"."GR_QPValues" FOR EACH ROW EXECUTE FUNCTION upd_timestamp();;

DELIMITER ;

DROP TABLE IF EXISTS "Gauges";
CREATE TABLE "public"."Gauges" (
    "ID" double precision NOT NULL,
    "PARAMETER" character varying(50) NOT NULL,
    CONSTRAINT "Gauges_pkey" PRIMARY KEY ("ID")
) WITH (oids = false);

CREATE INDEX "Gauges_PARAMETER" ON "public"."Gauges" USING btree ("PARAMETER");


DROP TABLE IF EXISTS "T_GR_QPValues";
CREATE TABLE "public"."T_GR_QPValues" (
    "COILID" character varying(254) NOT NULL,
    "TILEID" double precision NOT NULL,
    "MID" double precision NOT NULL,
    "PLANT" double precision NOT NULL,
    "SIDE" character varying(255) NOT NULL,
    "MEAN" double precision NOT NULL,
    "MAX" double precision NOT NULL,
    "MIN" double precision NOT NULL,
    "COUNT" double precision NOT NULL,
    "tstamp" timestamp DEFAULT now()
) WITH (oids = false);


DROP TABLE IF EXISTS "T_V_Coils";
CREATE TABLE "public"."T_V_Coils" (
    "id" integer NOT NULL,
    "SID" character varying(254) NOT NULL,
    "PLANT" double precision,
    "NAME" character varying(254),
    "STARTTIME" date,
    "MATERIAL" double precision,
    "LENGTH" double precision,
    "WIDTH" double precision,
    "THICK" double precision,
    "WEIGHT" double precision,
    "BASEPID" double precision,
    "CLASSLABEL" character varying(3) NOT NULL,
    "Label" integer NOT NULL,
    "ZnMin" integer,
    "ZnMax" integer,
    "tstamp" timestamp DEFAULT now()
) WITH (oids = false);


DROP TABLE IF EXISTS "V_Coils";
CREATE TABLE "public"."V_Coils" (
    "id" integer DEFAULT GENERATED BY DEFAULT AS IDENTITY NOT NULL,
    "SID" character varying(254) NOT NULL,
    "PLANT" double precision,
    "NAME" character varying(254),
    "STARTTIME" date,
    "MATERIAL" double precision,
    "LENGTH" double precision,
    "WIDTH" double precision,
    "THICK" double precision,
    "WEIGHT" double precision,
    "BASEPID" double precision,
    "CLASSLABEL" character varying(3),
    "Label" integer,
    "ZnMin" integer,
    "ZnMax" integer,
    "tstamp" timestamp DEFAULT now() NOT NULL,
    CONSTRAINT "V_CSID_idx" UNIQUE ("SID"),
    CONSTRAINT "V_Coils_SID_idx" UNIQUE ("SID"),
    CONSTRAINT "V_Coils_pkey" PRIMARY KEY ("id")
) WITH (oids = false);

CREATE INDEX "V_Coils_Label_idx" ON "public"."V_Coils" USING btree ("Label");


DELIMITER ;;

CREATE TRIGGER "t_name" BEFORE UPDATE ON "public"."V_Coils" FOR EACH ROW EXECUTE FUNCTION upd_timestamp();;

DELIMITER ;

DROP TABLE IF EXISTS "assessments";
DROP SEQUENCE IF EXISTS assessments_id_seq;
CREATE SEQUENCE assessments_id_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 4 CACHE 1;

CREATE TABLE "public"."assessments" (
    "id" integer DEFAULT nextval('assessments_id_seq') NOT NULL,
    "COILID" bigint NOT NULL,
    "MODELNAME" character varying(250) NOT NULL,
    "LABEL" integer NOT NULL,
    "ts" timestamp DEFAULT now() NOT NULL,
    "Confidence" real,
    CONSTRAINT "ass_unq_idx" UNIQUE ("COILID", "MODELNAME"),
    CONSTRAINT "assessments_pkey" PRIMARY KEY ("id")
) WITH (oids = false);


DROP TABLE IF EXISTS "feedback";
DROP SEQUENCE IF EXISTS feedback_id_seq;
CREATE SEQUENCE feedback_id_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 1 CACHE 1;

CREATE TABLE "public"."feedback" (
    "id" integer DEFAULT nextval('feedback_id_seq') NOT NULL,
    "COILID" character varying(50) NOT NULL,
    "Label" integer NOT NULL,
    "Descrip" character varying(255) NOT NULL,
    "ts" timestamp DEFAULT now() NOT NULL,
    CONSTRAINT "feedback_COILID" UNIQUE ("COILID"),
    CONSTRAINT "feedback_pkey" PRIMARY KEY ("id")
) WITH (oids = false);


DROP TABLE IF EXISTS "models";
DROP SEQUENCE IF EXISTS models_modelid_seq;
CREATE SEQUENCE models_modelid_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 7 CACHE 1;

CREATE TABLE "public"."models" (
    "modelid" integer DEFAULT nextval('models_modelid_seq') NOT NULL,
    "name" character varying(250) NOT NULL,
    "ts" timestamp DEFAULT now() NOT NULL,
    "type" character varying(50),
    "valid_until" timestamp,
    CONSTRAINT "models_name" UNIQUE ("name"),
    CONSTRAINT "models_pkey" PRIMARY KEY ("modelid")
) WITH (oids = false);


DROP TABLE IF EXISTS "mperformance";
DROP SEQUENCE IF EXISTS mperformance_id_seq;
CREATE SEQUENCE mperformance_id_seq INCREMENT 1 MINVALUE 1 MAXVALUE 2147483647 START 6 CACHE 1;

CREATE TABLE "public"."mperformance" (
    "id" integer DEFAULT nextval('mperformance_id_seq') NOT NULL,
    "mname" character varying(250) NOT NULL,
    "from_ass" date,
    "to_ass" date NOT NULL,
    "num_oper" integer,
    "ok_percentage" real,
    "ts" timestamp DEFAULT now() NOT NULL,
    "numc_assessed" integer,
    CONSTRAINT "mperformance_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "unique_perf" UNIQUE ("mname", "from_ass", "to_ass")
) WITH (oids = false);


-- 2024-02-14 22:11:29.028823+00